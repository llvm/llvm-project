//===- RISCVVEmitter.cpp - Generate riscv_vector.h for use with clang -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting riscv_vector.h and
// riscv_vector_generic.h, which includes a declaration and definition of each
// intrinsic fucntions specified in https://github.com/riscv/rvv-intrinsic-doc.
//
// See also the documentation in include/clang/Basic/riscv_vector.td.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <numeric>

using namespace llvm;
using BasicType = char;
using VScaleVal = Optional<unsigned>;

namespace {

// Exponential LMUL
class LMULType {
private:
  int Log2LMUL;

public:
  LMULType(int Log2LMUL);
  // Return the C/C++ string representation of LMUL
  std::string str() const;
  Optional<unsigned> getScale(unsigned ElementBitwidth) const;
  LMULType &operator*=(unsigned RHS);
};

// This class is compact representation of a valid and invalid RVVType.
class RVVType {
  enum ScalarTypeKind : uint32_t {
    Void,
    Size_t,
    Ptrdiff_t,
    Boolean,
    SignedInteger,
    UnsignedInteger,
    Float,
    Invalid,
  };
  BasicType BT;
  ScalarTypeKind ScalarType = Invalid;
  LMULType LMUL;
  bool IsPointer = false;
  // IsConstant indices are "int", but have the constant expression.
  bool IsImmediate = false;
  // Const qualifier for pointer to const object or object of const type.
  bool IsConstant = false;
  unsigned ElementBitwidth = 0;
  VScaleVal Scale = 0;
  bool Valid;

  std::string BuiltinStr;
  std::string ClangBuiltinStr;
  std::string Str;
  std::string ShortStr;

public:
  RVVType() : RVVType(BasicType(), 0, StringRef()) {}
  RVVType(BasicType BT, int Log2LMUL, StringRef prototype);

  // Return the string representation of a type, which is an encoded string for
  // passing to the BUILTIN() macro in Builtins.def.
  const std::string &getBuiltinStr() const { return BuiltinStr; }

  // Return the clang buitlin type for RVV vector type which are used in the
  // riscv_vector.h header file.
  const std::string &getClangBuiltinStr() const { return ClangBuiltinStr; }

  // Return the C/C++ string representation of a type for use in the
  // riscv_vector.h header file.
  const std::string &getTypeStr() const { return Str; }

  // Return the short name of a type for C/C++ name suffix.
  const std::string &getShortStr() const { return ShortStr; }

  bool isValid() const { return Valid; }
  bool isScalar() const { return Scale.hasValue() && Scale.getValue() == 0; }
  bool isVector() const { return Scale.hasValue() && Scale.getValue() != 0; }
  bool isFloat() const { return ScalarType == ScalarTypeKind::Float; }
  bool isSignedInteger() const {
    return ScalarType == ScalarTypeKind::SignedInteger;
  }
  bool isFloatVector(unsigned Width) const {
    return isVector() && isFloat() && ElementBitwidth == Width;
  }

private:
  // Verify RVV vector type and set Valid.
  bool verifyType() const;

  // Creates a type based on basic types of TypeRange
  void applyBasicType();

  // Applies a prototype modifier to the current type. The result maybe an
  // invalid type.
  void applyModifier(StringRef prototype);

  // Compute and record a string for legal type.
  void initBuiltinStr();
  // Compute and record a builtin RVV vector type string.
  void initClangBuiltinStr();
  // Compute and record a type string for used in the header.
  void initTypeStr();
  // Compute and record a short name of a type for C/C++ name suffix.
  void initShortStr();
};

using RVVTypePtr = RVVType *;
using RVVTypes = std::vector<RVVTypePtr>;

enum RISCVExtension : uint8_t {
  Basic = 0,
  F = 1 << 1,
  D = 1 << 2,
  Zfh = 1 << 3
};

// TODO refactor RVVIntrinsic class design after support all intrinsic
// combination. This represents an instantiation of an intrinsic with a
// particular type and prototype
class RVVIntrinsic {

private:
  std::string Name; // Builtin name
  std::string MangledName;
  std::string IRName;
  bool HasSideEffects;
  bool HasMaskedOffOperand;
  bool HasVL;
  bool HasGeneric;
  RVVTypePtr OutputType; // Builtin output type
  RVVTypes InputTypes;   // Builtin input types
  // The types we use to obtain the specific LLVM intrinsic. They are index of
  // InputTypes. -1 means the return type.
  std::vector<int64_t> IntrinsicTypes;
  // C/C++ intrinsic operand order is different to builtin operand order. Record
  // the mapping of InputTypes index.
  SmallVector<unsigned> CTypeOrder;
  uint8_t RISCVExtensions = 0;

public:
  RVVIntrinsic(StringRef Name, StringRef Suffix, StringRef MangledName,
               StringRef IRName, bool HasSideEffects, bool IsMask,
               bool HasMaskedOffOperand, bool HasVL, bool HasGeneric,
               const RVVTypes &Types,
               const std::vector<int64_t> &RVVIntrinsicTypes);
  ~RVVIntrinsic() = default;

  StringRef getName() const { return Name; }
  StringRef getMangledName() const { return MangledName; }
  bool hasSideEffects() const { return HasSideEffects; }
  bool hasMaskedOffOperand() const { return HasMaskedOffOperand; }
  bool hasVL() const { return HasVL; }
  bool hasGeneric() const { return HasGeneric; }
  size_t getNumOperand() const { return InputTypes.size(); }
  StringRef getIRName() const { return IRName; }
  uint8_t getRISCVExtensions() const { return RISCVExtensions; }

  // Return the type string for a BUILTIN() macro in Builtins.def.
  std::string getBuiltinTypeStr() const;

  // Emit the code block for switch body in EmitRISCVBuiltinExpr, it should
  // init the RVVIntrinsic ID and IntrinsicTypes.
  void emitCodeGenSwitchBody(raw_ostream &o) const;

  // Emit the macros for mapping C/C++ intrinsic function to builtin functions.
  void emitIntrinsicMacro(raw_ostream &o) const;

  // Emit the mangled function definition.
  void emitMangledFuncDef(raw_ostream &o) const;
};

class RVVEmitter {
private:
  RecordKeeper &Records;
  // Concat BasicType, LMUL and Proto as key
  StringMap<RVVType> LegalTypes;
  StringSet<> IllegalTypes;

public:
  RVVEmitter(RecordKeeper &R) : Records(R) {}

  /// Emit riscv_vector.h
  void createHeader(raw_ostream &o);

  /// Emit riscv_generic.h
  void createGenericHeader(raw_ostream &o);

  /// Emit all the __builtin prototypes and code needed by Sema.
  void createBuiltins(raw_ostream &o);

  /// Emit all the information needed to map builtin -> LLVM IR intrinsic.
  void createCodeGen(raw_ostream &o);

private:
  /// Create all intrinsics and add them to \p Out
  void createRVVIntrinsics(std::vector<std::unique_ptr<RVVIntrinsic>> &Out);
  /// Compute output and input types by applying different config (basic type
  /// and LMUL with type transformers). It also record result of type in legal
  /// or illegal set to avoid compute the  same config again. The result maybe
  /// have illegal RVVType.
  Optional<RVVTypes> computeTypes(BasicType BT, int Log2LMUL,
                                  ArrayRef<std::string> PrototypeSeq);
  Optional<RVVTypePtr> computeType(BasicType BT, int Log2LMUL, StringRef Proto);

  /// Emit Acrh predecessor definitions and body
  void emitArchMacroAndBody(
      std::vector<std::unique_ptr<RVVIntrinsic>> &Defs, raw_ostream &o,
      std::function<void(raw_ostream &, const RVVIntrinsic &)>);

  // Emit the architecture preprocessor definitions. Return true when emits
  // non-empty string.
  bool emitExtDefStr(uint8_t Extensions, raw_ostream &o);
};

} // namespace

//===----------------------------------------------------------------------===//
// Type implementation
//===----------------------------------------------------------------------===//

LMULType::LMULType(int NewLog2LMUL) {
  // Check Log2LMUL is -3, -2, -1, 0, 1, 2, 3
  assert(NewLog2LMUL <= 3 && NewLog2LMUL >= -3 && "Bad LMUL number!");
  Log2LMUL = NewLog2LMUL;
}

std::string LMULType::str() const {
  if (Log2LMUL < 0)
    return "mf" + utostr(1ULL << (-Log2LMUL));
  return "m" + utostr(1ULL << Log2LMUL);
}

VScaleVal LMULType::getScale(unsigned ElementBitwidth) const {
  int Log2ScaleResult = 0;
  switch (ElementBitwidth) {
  default:
    break;
  case 8:
    Log2ScaleResult = Log2LMUL + 3;
    break;
  case 16:
    Log2ScaleResult = Log2LMUL + 2;
    break;
  case 32:
    Log2ScaleResult = Log2LMUL + 1;
    break;
  case 64:
    Log2ScaleResult = Log2LMUL;
    break;
  }
  // Illegal vscale result would be less than 1
  if (Log2ScaleResult < 0)
    return None;
  return 1 << Log2ScaleResult;
}

LMULType &LMULType::operator*=(uint32_t RHS) {
  assert(isPowerOf2_32(RHS));
  this->Log2LMUL = this->Log2LMUL + Log2_32(RHS);
  return *this;
}

RVVType::RVVType(BasicType BT, int Log2LMUL, StringRef prototype)
    : BT(BT), LMUL(LMULType(Log2LMUL)) {
  applyBasicType();
  applyModifier(prototype);
  Valid = verifyType();
  if (Valid) {
    initBuiltinStr();
    initTypeStr();
    if (isVector()) {
      initClangBuiltinStr();
      initShortStr();
    }
  }
}

// clang-format off
// boolean type are encoded the ratio of n (SEW/LMUL)
// SEW/LMUL | 1         | 2         | 4         | 8        | 16        | 32        | 64
// c type   | vbool64_t | vbool32_t | vbool16_t | vbool8_t | vbool4_t  | vbool2_t  | vbool1_t
// IR type  | nxv1i1    | nxv2i1    | nxv4i1    | nxv8i1   | nxv16i1   | nxv32i1   | nxv64i1

// type\lmul | 1/8    | 1/4      | 1/2     | 1       | 2        | 4        | 8
// --------  |------  | -------- | ------- | ------- | -------- | -------- | --------
// i64       | N/A    | N/A      | N/A     | nxv1i64 | nxv2i64  | nxv4i64  | nxv8i64
// i32       | N/A    | N/A      | nxv1i32 | nxv2i32 | nxv4i32  | nxv8i32  | nxv16i32
// i16       | N/A    | nxv1i16  | nxv2i16 | nxv4i16 | nxv8i16  | nxv16i16 | nxv32i16
// i8        | nxv1i8 | nxv2i8   | nxv4i8  | nxv8i8  | nxv16i8  | nxv32i8  | nxv64i8
// double    | N/A    | N/A      | N/A     | nxv1f64 | nxv2f64  | nxv4f64  | nxv8f64
// float     | N/A    | N/A      | nxv1f32 | nxv2f32 | nxv4f32  | nxv8f32  | nxv16f32
// half      | N/A    | nxv1f16  | nxv2f16 | nxv4f16 | nxv8f16  | nxv16f16 | nxv32f16
// clang-format on

bool RVVType::verifyType() const {
  if (isScalar())
    return true;
  if (!Scale.hasValue())
    return false;
  if (isFloat() && ElementBitwidth == 8)
    return false;
  unsigned V = Scale.getValue();
  switch (ElementBitwidth) {
  case 1:
  case 8:
    // Check Scale is 1,2,4,8,16,32,64
    return (V <= 64 && isPowerOf2_32(V));
  case 16:
    // Check Scale is 1,2,4,8,16,32
    return (V <= 32 && isPowerOf2_32(V));
  case 32:
    // Check Scale is 1,2,4,8,16
    return (V <= 16 && isPowerOf2_32(V));
  case 64:
    // Check Scale is 1,2,4,8
    return (V <= 8 && isPowerOf2_32(V));
  }
  return false;
}

void RVVType::initBuiltinStr() {
  assert(isValid() && "RVVType is invalid");
  switch (ScalarType) {
  case ScalarTypeKind::Void:
    BuiltinStr = "v";
    return;
  case ScalarTypeKind::Size_t:
    BuiltinStr = "z";
    if (IsImmediate)
      BuiltinStr = "I" + BuiltinStr;
    return;
  case ScalarTypeKind::Ptrdiff_t:
    BuiltinStr = "Y";
    return;
  case ScalarTypeKind::Boolean:
    assert(ElementBitwidth == 1);
    BuiltinStr += "b";
    break;
  case ScalarTypeKind::SignedInteger:
  case ScalarTypeKind::UnsignedInteger:
    switch (ElementBitwidth) {
    case 8:
      BuiltinStr += "c";
      break;
    case 16:
      BuiltinStr += "s";
      break;
    case 32:
      BuiltinStr += "i";
      break;
    case 64:
      BuiltinStr += "Wi";
      break;
    default:
      llvm_unreachable("Unhandled ElementBitwidth!");
    }
    if (isSignedInteger())
      BuiltinStr = "S" + BuiltinStr;
    else
      BuiltinStr = "U" + BuiltinStr;
    break;
  case ScalarTypeKind::Float:
    switch (ElementBitwidth) {
    case 16:
      BuiltinStr += "h";
      break;
    case 32:
      BuiltinStr += "f";
      break;
    case 64:
      BuiltinStr += "d";
      break;
    default:
      llvm_unreachable("Unhandled ElementBitwidth!");
    }
    break;
  default:
    llvm_unreachable("ScalarType is invalid!");
  }
  if (IsImmediate)
    BuiltinStr = "I" + BuiltinStr;
  if (isScalar()) {
    if (IsConstant)
      BuiltinStr += "C";
    if (IsPointer)
      BuiltinStr += "*";
    return;
  }
  BuiltinStr = "q" + utostr(Scale.getValue()) + BuiltinStr;
}

void RVVType::initClangBuiltinStr() {
  assert(isValid() && "RVVType is invalid");
  assert(isVector() && "Handle Vector type only");

  ClangBuiltinStr = "__rvv_";
  switch (ScalarType) {
  case ScalarTypeKind::Boolean:
    ClangBuiltinStr += "bool" + utostr(64 / Scale.getValue()) + "_t";
    return;
  case ScalarTypeKind::Float:
    ClangBuiltinStr += "float";
    break;
  case ScalarTypeKind::SignedInteger:
    ClangBuiltinStr += "int";
    break;
  case ScalarTypeKind::UnsignedInteger:
    ClangBuiltinStr += "uint";
    break;
  default:
    llvm_unreachable("ScalarTypeKind is invalid");
  }
  ClangBuiltinStr += utostr(ElementBitwidth) + LMUL.str() + "_t";
}

void RVVType::initTypeStr() {
  assert(isValid() && "RVVType is invalid");

  if (IsConstant)
    Str += "const ";

  auto getTypeString = [&](StringRef TypeStr) {
    if (isScalar())
      return Twine(TypeStr + Twine(ElementBitwidth) + "_t").str();
    return Twine("v" + TypeStr + Twine(ElementBitwidth) + LMUL.str() + "_t")
        .str();
  };

  switch (ScalarType) {
  case ScalarTypeKind::Void:
    Str = "void";
    return;
  case ScalarTypeKind::Size_t:
    Str = "size_t";
    return;
  case ScalarTypeKind::Ptrdiff_t:
    Str = "ptrdiff_t";
    return;
  case ScalarTypeKind::Boolean:
    if (isScalar())
      Str += "bool";
    else
      // Vector bool is special case, the formulate is
      // `vbool<N>_t = MVT::nxv<64/N>i1` ex. vbool16_t = MVT::4i1
      Str += "vbool" + utostr(64 / Scale.getValue()) + "_t";
    break;
  case ScalarTypeKind::Float:
    if (isScalar()) {
      if (ElementBitwidth == 64)
        Str += "double";
      else if (ElementBitwidth == 32)
        Str += "float";
      assert((ElementBitwidth == 32 || ElementBitwidth == 64) &&
             "Unhandled floating type");
    } else
      Str += getTypeString("float");
    break;
  case ScalarTypeKind::SignedInteger:
    Str += getTypeString("int");
    break;
  case ScalarTypeKind::UnsignedInteger:
    Str += getTypeString("uint");
    break;
  default:
    llvm_unreachable("ScalarType is invalid!");
  }
  if (IsPointer)
    Str += " *";
}

void RVVType::initShortStr() {
  assert(isVector() && "only handle vector type");
  switch (ScalarType) {
  case ScalarTypeKind::Boolean:
    ShortStr = "b" + utostr(64 / Scale.getValue());
    break;
  case ScalarTypeKind::Float:
    ShortStr = "f" + utostr(ElementBitwidth) + LMUL.str();
    break;
  case ScalarTypeKind::SignedInteger:
    ShortStr = "i" + utostr(ElementBitwidth) + LMUL.str();
    break;
  case ScalarTypeKind::UnsignedInteger:
    ShortStr = "u" + utostr(ElementBitwidth) + LMUL.str();
    break;
  default:
    llvm_unreachable("Unhandled case!");
  }
}

void RVVType::applyBasicType() {
  switch (BT) {
  case 'c':
    ElementBitwidth = 8;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case 's':
    ElementBitwidth = 16;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case 'i':
    ElementBitwidth = 32;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case 'l':
    ElementBitwidth = 64;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case 'h':
    ElementBitwidth = 16;
    ScalarType = ScalarTypeKind::Float;
    break;
  case 'f':
    ElementBitwidth = 32;
    ScalarType = ScalarTypeKind::Float;
    break;
  case 'd':
    ElementBitwidth = 64;
    ScalarType = ScalarTypeKind::Float;
    break;
  default:
    PrintFatalError("Unhandled type code!");
  }
  assert(ElementBitwidth != 0 && "Bad element bitwidth!");
}

void RVVType::applyModifier(StringRef Transformer) {
  if (Transformer.empty())
    return;
  // Handle primitive type transformer
  switch (Transformer.back()) {
  case 'e':
    Scale = 0;
    break;
  case 'v':
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case 'w':
    ElementBitwidth *= 2;
    LMUL *= 2;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case 'q':
    ElementBitwidth *= 4;
    LMUL *= 4;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case 'o':
    ElementBitwidth *= 8;
    LMUL *= 8;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case 'm':
    ScalarType = ScalarTypeKind::Boolean;
    Scale = LMUL.getScale(ElementBitwidth);
    ElementBitwidth = 1;
    break;
  case '0':
    ScalarType = ScalarTypeKind::Void;
    break;
  case 'z':
    ScalarType = ScalarTypeKind::Size_t;
    break;
  case 't':
    ScalarType = ScalarTypeKind::Ptrdiff_t;
    break;
  case 'c': // uint8_t
    ScalarType = ScalarTypeKind::UnsignedInteger;
    ElementBitwidth = 8;
    Scale = 0;
    break;
  default:
    PrintFatalError("Illegal primitive type transformers!");
  }
  Transformer = Transformer.drop_back();

  // Compute type transformers
  for (char I : Transformer) {
    switch (I) {
    case 'P':
      if (IsConstant)
        PrintFatalError("'P' transformer cannot be used after 'C'");
      if (IsPointer)
        PrintFatalError("'P' transformer cannot be used twice");
      IsPointer = true;
      break;
    case 'C':
      if (IsConstant)
        PrintFatalError("'C' transformer cannot be used twice");
      IsConstant = true;
      break;
    case 'K':
      IsImmediate = true;
      break;
    case 'U':
      ScalarType = ScalarTypeKind::UnsignedInteger;
      break;
    case 'I':
      ScalarType = ScalarTypeKind::SignedInteger;
      break;
    case 'F':
      ScalarType = ScalarTypeKind::Float;
      break;
    case 'S':
      LMUL = LMULType(0);
      // Update ElementBitwidth need to update Scale too.
      Scale = LMUL.getScale(ElementBitwidth);
      break;
    default:
      PrintFatalError("Illegal non-primitive type transformer!");
    }
  }
}

//===----------------------------------------------------------------------===//
// RVVIntrinsic implementation
//===----------------------------------------------------------------------===//
RVVIntrinsic::RVVIntrinsic(StringRef NewName, StringRef Suffix,
                           StringRef NewMangledName, StringRef IRName,
                           bool HasSideEffects, bool IsMask,
                           bool HasMaskedOffOperand, bool HasVL,
                           bool HasGeneric, const RVVTypes &OutInTypes,
                           const std::vector<int64_t> &NewIntrinsicTypes)
    : IRName(IRName), HasSideEffects(HasSideEffects),
      HasMaskedOffOperand(HasMaskedOffOperand), HasVL(HasVL),
      HasGeneric(HasGeneric) {

  // Init Name and MangledName
  Name = NewName.str();
  if (NewMangledName.empty())
    MangledName = NewName.split("_").first.str();
  else
    MangledName = NewMangledName.str();
  if (!Suffix.empty())
    Name += "_" + Suffix.str();
  if (IsMask) {
    Name += "_m";
    MangledName += "_m";
  }
  // Init RISC-V extensions
  for (const auto &T : OutInTypes) {
    if (T->isFloatVector(16))
      RISCVExtensions |= RISCVExtension::Zfh;
    else if (T->isFloatVector(32))
      RISCVExtensions |= RISCVExtension::F;
    else if (T->isFloatVector(64))
      RISCVExtensions |= RISCVExtension::D;
  }

  // Init OutputType and InputTypes
  OutputType = OutInTypes[0];
  InputTypes.assign(OutInTypes.begin() + 1, OutInTypes.end());
  CTypeOrder.resize(InputTypes.size());
  std::iota(CTypeOrder.begin(), CTypeOrder.end(), 0);
  if (IsMask) {
    if (HasVL)
      // Builtin type order: op0, op1, ..., mask, vl
      // C type order: mask, op0, op1, ..., vl
      std::rotate(CTypeOrder.begin(), CTypeOrder.end() - 2,
                  CTypeOrder.end() - 1);
    else
      // Builtin type order: op0, op1, ..., mask
      // C type order: mask, op0, op1, ...,
      std::rotate(CTypeOrder.begin(), CTypeOrder.end() - 1, CTypeOrder.end());
  }
  // IntrinsicTypes is nonmasked version index. Need to update it
  // if there is maskedoff operand (It is always in first operand).
  IntrinsicTypes = NewIntrinsicTypes;
  if (IsMask && HasMaskedOffOperand) {
    for (auto &I : IntrinsicTypes) {
      if (I >= 0)
        I += 1;
    }
  }
}

std::string RVVIntrinsic::getBuiltinTypeStr() const {
  std::string S;
  S += OutputType->getBuiltinStr();
  for (const auto &T : InputTypes) {
    S += T->getBuiltinStr();
  }
  return S;
}

void RVVIntrinsic::emitCodeGenSwitchBody(raw_ostream &OS) const {
  OS << "  ID = Intrinsic::riscv_" + getIRName() + ";\n";
  OS << "  IntrinsicTypes = {";
  ListSeparator LS;
  for (const auto &Idx : IntrinsicTypes) {
    if (Idx == -1)
      OS << LS << "ResultType";
    else
      OS << LS << "Ops[" << Idx << "]->getType()";
  }

  // VL could be i64 or i32, need to encode it in IntrinsicTypes. VL is
  // always last operand.
  if (hasVL())
    OS << ", Ops[" << getNumOperand() - 1 << "]->getType()";
  OS << "};\n";
  OS << "  break;\n";
}

void RVVIntrinsic::emitIntrinsicMacro(raw_ostream &OS) const {
  OS << "#define " << getName() << "(";
  if (getNumOperand() > 0) {
    ListSeparator LS;
    for (const auto &I : CTypeOrder)
      OS << LS << "op" << I;
  }
  OS << ") \\\n";
  OS << "__builtin_rvv_" << getName() << "(";
  if (getNumOperand() > 0) {
    ListSeparator LS;
    for (unsigned i = 0; i < InputTypes.size(); ++i)
      OS << LS << "(" << InputTypes[i]->getTypeStr() << ")(op" << i << ")";
  }
  OS << ")\n";
}

void RVVIntrinsic::emitMangledFuncDef(raw_ostream &OS) const {
  OS << OutputType->getTypeStr() << " " << getMangledName() << "(";
  // Emit function arguments
  if (getNumOperand() > 0) {
    ListSeparator LS;
    for (unsigned i = 0; i < CTypeOrder.size(); ++i)
      OS << LS << InputTypes[CTypeOrder[i]]->getTypeStr() << " op" << i;
  }
  OS << "){\n";
  OS << "  return " << getName() << "(";
  // Emit parameter variables
  if (getNumOperand() > 0) {
    ListSeparator LS;
    for (unsigned i = 0; i < CTypeOrder.size(); ++i)
      OS << LS << "op" << i;
  }
  OS << ");\n";
  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// RVVEmitter implementation
//===----------------------------------------------------------------------===//
void RVVEmitter::createHeader(raw_ostream &OS) {

  OS << "/*===---- riscv_vector.h - RISC-V V-extension RVVIntrinsics "
        "-------------------===\n"
        " *\n"
        " *\n"
        " * Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n"
        " * See https://llvm.org/LICENSE.txt for license information.\n"
        " * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        " *\n"
        " *===-----------------------------------------------------------------"
        "------===\n"
        " */\n\n";

  OS << "#ifndef __RISCV_VECTOR_H\n";
  OS << "#define __RISCV_VECTOR_H\n\n";

  OS << "#include <stdint.h>\n";
  OS << "#include <stddef.h>\n\n";

  OS << "#ifndef __riscv_vector\n";
  OS << "#error \"Vector intrinsics require the vector extension.\"\n";
  OS << "#endif\n\n";

  OS << "#ifdef __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#endif\n\n";

  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  auto printType = [&](auto T) {
    OS << "typedef " << T->getClangBuiltinStr() << " " << T->getTypeStr()
       << ";\n";
  };

  constexpr int Log2LMULs[] = {-3, -2, -1, 0, 1, 2, 3};
  // Print RVV boolean types.
  for (int Log2LMUL : Log2LMULs) {
    auto T = computeType('c', Log2LMUL, "m");
    if (T.hasValue())
      printType(T.getValue());
  }
  // Print RVV int/float types.
  for (char I : StringRef("csil")) {
    for (int Log2LMUL : Log2LMULs) {
      auto T = computeType(I, Log2LMUL, "v");
      if (T.hasValue()) {
        printType(T.getValue());
        auto UT = computeType(I, Log2LMUL, "Uv");
        printType(UT.getValue());
      }
    }
  }
  OS << "#if defined(__riscv_zfh)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = computeType('h', Log2LMUL, "v");
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_f)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = computeType('f', Log2LMUL, "v");
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_d)\n";
  for (int ELMul : Log2LMULs) {
    auto T = computeType('d', ELMul, "v");
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n\n";

  // Print intrinsic functions with macro
  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    Inst.emitIntrinsicMacro(OS);
  });

  OS << "\n#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif // __riscv_vector\n";
  OS << "#endif // __RISCV_VECTOR_H\n";
}

void RVVEmitter::createGenericHeader(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  OS << "#include <riscv_vector.h>\n\n";
  // Print intrinsic functions macro
  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    if (!Inst.hasGeneric())
      return;
    OS << "static inline __attribute__((__always_inline__, __nodebug__, "
          "__overloadable__))\n";
    Inst.emitMangledFuncDef(OS);
  });
}

void RVVEmitter::createBuiltins(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  OS << "#if defined(TARGET_BUILTIN) && !defined(RISCVV_BUILTIN)\n";
  OS << "#define RISCVV_BUILTIN(ID, TYPE, ATTRS) TARGET_BUILTIN(ID, TYPE, "
        "ATTRS, \"experimental-v\")\n";
  OS << "#endif\n";
  for (auto &Def : Defs) {
    OS << "RISCVV_BUILTIN(__builtin_rvv_" << Def->getName() << ",\""
       << Def->getBuiltinTypeStr() << "\", ";
    if (!Def->hasSideEffects())
      OS << "\"n\")\n";
    else
      OS << "\"\")\n";
  }
  OS << "#undef RISCVV_BUILTIN\n";
}

void RVVEmitter::createCodeGen(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  // The same intrinsic IR name has the same switch body.
  std::stable_sort(Defs.begin(), Defs.end(),
                   [](const std::unique_ptr<RVVIntrinsic> &A,
                      const std::unique_ptr<RVVIntrinsic> &B) {
                     return A->getIRName() < B->getIRName();
                   });
  // Print switch body when the ir name changes from previous iteration.
  RVVIntrinsic *PrevDef = Defs.begin()->get();
  for (auto &Def : Defs) {
    StringRef CurIRName = Def->getIRName();
    if (CurIRName != PrevDef->getIRName()) {
      PrevDef->emitCodeGenSwitchBody(OS);
    }
    PrevDef = Def.get();
    OS << "case RISCV::BI__builtin_rvv_" << Def->getName() << ":\n";
  }
  Defs.back()->emitCodeGenSwitchBody(OS);
  OS << "\n";
}

void RVVEmitter::createRVVIntrinsics(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Out) {

  std::vector<Record *> RV = Records.getAllDerivedDefinitions("RVVBuiltin");
  for (auto *R : RV) {
    StringRef Name = R->getValueAsString("Name");
    StringRef Suffix = R->getValueAsString("Suffix");
    StringRef MangledName = R->getValueAsString("MangledName");
    StringRef Prototypes = R->getValueAsString("Prototype");
    StringRef TypeRange = R->getValueAsString("TypeRange");
    bool HasMask = R->getValueAsBit("HasMask");
    bool HasMaskedOffOperand = R->getValueAsBit("HasMaskedOffOperand");
    bool HasVL = R->getValueAsBit("HasVL");
    bool HasGeneric = R->getValueAsBit("HasGeneric");
    bool HasSideEffects = R->getValueAsBit("HasSideEffects");
    std::vector<int64_t> Log2LMULList = R->getValueAsListOfInts("Log2LMUL");
    std::vector<int64_t> IntrinsicTypes =
        R->getValueAsListOfInts("IntrinsicTypes");
    StringRef IRName = R->getValueAsString("IRName");
    StringRef IRNameMask = R->getValueAsString("IRNameMask");

    // Parse prototype and create a list of primitive type with transformers
    // (operand) in ProtoSeq. ProtoSeq[0] is output operand.
    SmallVector<std::string, 8> ProtoSeq;
    const StringRef Primaries("evwqom0ztc");
    while (!Prototypes.empty()) {
      auto Idx = Prototypes.find_first_of(Primaries);
      assert(Idx != StringRef::npos);
      ProtoSeq.push_back(Prototypes.slice(0, Idx + 1).str());
      Prototypes = Prototypes.drop_front(Idx + 1);
    }

    // Compute Builtin types
    SmallVector<std::string, 8> ProtoMaskSeq = ProtoSeq;
    if (HasMask) {
      // If HasMask, append 'm' to last operand.
      ProtoMaskSeq.push_back("m");
      // If HasMaskedOffOperand, insert result type as first input operand.
      if (HasMaskedOffOperand)
        ProtoMaskSeq.insert(ProtoMaskSeq.begin() + 1, ProtoSeq[0]);
    }
    // If HasVL, append 'z' to last operand
    if (HasVL) {
      ProtoSeq.push_back("z");
      ProtoMaskSeq.push_back("z");
    }

    // Create intrinsics for each type and LMUL.
    for (char I : TypeRange) {
      for (int Log2LMUL : Log2LMULList) {
        Optional<RVVTypes> Types = computeTypes(I, Log2LMUL, ProtoSeq);
        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue())
          continue;

        auto SuffixStr =
            computeType(I, Log2LMUL, Suffix).getValue()->getShortStr();
        // Create a non-mask intrinsic.
        Out.push_back(std::make_unique<RVVIntrinsic>(
            Name, SuffixStr, MangledName, IRName, HasSideEffects,
            /*IsMask=*/false, /*HasMaskedOffOperand=*/false, HasVL, HasGeneric,
            Types.getValue(), IntrinsicTypes));
        if (HasMask) {
          // Create a mask intrinsic
          Optional<RVVTypes> MaskTypes =
              computeTypes(I, Log2LMUL, ProtoMaskSeq);
          Out.push_back(std::make_unique<RVVIntrinsic>(
              Name, SuffixStr, MangledName, IRNameMask, HasSideEffects,
              /*IsMask=*/true, HasMaskedOffOperand, HasVL, HasGeneric,
              MaskTypes.getValue(), IntrinsicTypes));
        }
      } // end for Log2LMUL
    }   // end for TypeRange
  }
}

Optional<RVVTypes>
RVVEmitter::computeTypes(BasicType BT, int Log2LMUL,
                         ArrayRef<std::string> PrototypeSeq) {
  RVVTypes Types;
  for (const std::string &Proto : PrototypeSeq) {
    auto T = computeType(BT, Log2LMUL, Proto);
    if (!T.hasValue())
      return llvm::None;
    // Record legal type index
    Types.push_back(T.getValue());
  }
  return Types;
}

Optional<RVVTypePtr> RVVEmitter::computeType(BasicType BT, int Log2LMUL,
                                             StringRef Proto) {
  std::string Idx = Twine(Twine(BT) + Twine(Log2LMUL) + Proto).str();
  // Search first
  auto It = LegalTypes.find(Idx);
  if (It != LegalTypes.end())
    return &(It->second);
  if (IllegalTypes.count(Idx))
    return llvm::None;
  // Compute type and record the result.
  RVVType T(BT, Log2LMUL, Proto);
  if (T.isValid()) {
    // Record legal type index and value.
    LegalTypes.insert({Idx, T});
    return &(LegalTypes[Idx]);
  }
  // Record illegal type index.
  IllegalTypes.insert(Idx);
  return llvm::None;
}

void RVVEmitter::emitArchMacroAndBody(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Defs, raw_ostream &OS,
    std::function<void(raw_ostream &, const RVVIntrinsic &)> PrintBody) {

  // The same extension include in the same arch guard marco.
  std::stable_sort(Defs.begin(), Defs.end(),
                   [](const std::unique_ptr<RVVIntrinsic> &A,
                      const std::unique_ptr<RVVIntrinsic> &B) {
                     return A->getRISCVExtensions() < B->getRISCVExtensions();
                   });
  uint8_t PrevExt = (*Defs.begin())->getRISCVExtensions();
  bool NeedEndif = emitExtDefStr(PrevExt, OS);
  for (auto &Def : Defs) {
    uint8_t CurExt = Def->getRISCVExtensions();
    if (CurExt != PrevExt) {
      if (NeedEndif)
        OS << "#endif\n\n";
      NeedEndif = emitExtDefStr(CurExt, OS);
      PrevExt = CurExt;
    }
    PrintBody(OS, *Def);
  }
  if (NeedEndif)
    OS << "#endif\n\n";
}

bool RVVEmitter::emitExtDefStr(uint8_t Extents, raw_ostream &OS) {
  if (Extents == RISCVExtension::Basic)
    return false;
  OS << "#if ";
  ListSeparator LS(" || ");
  if (Extents & RISCVExtension::F)
    OS << LS << "defined(__riscv_f)";
  if (Extents & RISCVExtension::D)
    OS << LS << "defined(__riscv_d)";
  if (Extents & RISCVExtension::Zfh)
    OS << LS << "defined(__riscv_zfh)";
  OS << "\n";
  return true;
}

namespace clang {
void EmitRVVHeader(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createHeader(OS);
}

void EmitRVVGenericHeader(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createGenericHeader(OS);
}

void EmitRVVBuiltins(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createBuiltins(OS);
}

void EmitRVVBuiltinCG(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createCodeGen(OS);
}

} // End namespace clang
