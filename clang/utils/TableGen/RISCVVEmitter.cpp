//===- RISCVVEmitter.cpp - Generate riscv_vector.h for use with clang -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting riscv_vector.h which
// includes a declaration and definition of each intrinsic functions specified
// in https://github.com/riscv/rvv-intrinsic-doc.
//
// See also the documentation in include/clang/Basic/riscv_vector.td.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/RISCVVIntrinsicUtils.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <numeric>

using namespace llvm;
using namespace llvm::RISCV;

namespace {
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

  /// Emit all the __builtin prototypes and code needed by Sema.
  void createBuiltins(raw_ostream &o);

  /// Emit all the information needed to map builtin -> LLVM IR intrinsic.
  void createCodeGen(raw_ostream &o);

  std::string getSuffixStr(char Type, int Log2LMUL, StringRef Prototypes);

private:
  /// Create all intrinsics and add them to \p Out
  void createRVVIntrinsics(std::vector<std::unique_ptr<RVVIntrinsic>> &Out);
  /// Print HeaderCode in RVVHeader Record to \p Out
  void printHeaderCode(raw_ostream &OS);
  /// Compute output and input types by applying different config (basic type
  /// and LMUL with type transformers). It also record result of type in legal
  /// or illegal set to avoid compute the  same config again. The result maybe
  /// have illegal RVVType.
  Optional<RVVTypes> computeTypes(BasicType BT, int Log2LMUL, unsigned NF,
                                  ArrayRef<std::string> PrototypeSeq);
  Optional<RVVTypePtr> computeType(BasicType BT, int Log2LMUL, StringRef Proto);

  /// Emit Acrh predecessor definitions and body, assume the element of Defs are
  /// sorted by extension.
  void emitArchMacroAndBody(
      std::vector<std::unique_ptr<RVVIntrinsic>> &Defs, raw_ostream &o,
      std::function<void(raw_ostream &, const RVVIntrinsic &)>);

  // Emit the architecture preprocessor definitions. Return true when emits
  // non-empty string.
  bool emitMacroRestrictionStr(RISCVPredefinedMacroT PredefinedMacros,
                               raw_ostream &o);
  // Slice Prototypes string into sub prototype string and process each sub
  // prototype string individually in the Handler.
  void parsePrototypes(StringRef Prototypes,
                       std::function<void(StringRef)> Handler);
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
    return llvm::None;
  return 1 << Log2ScaleResult;
}

void LMULType::MulLog2LMUL(int log2LMUL) { Log2LMUL += log2LMUL; }

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
  if (ScalarType == Invalid)
    return false;
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
    if (IsPointer)
      BuiltinStr += "*";
    return;
  case ScalarTypeKind::Ptrdiff_t:
    BuiltinStr = "Y";
    return;
  case ScalarTypeKind::UnsignedLong:
    BuiltinStr = "ULi";
    return;
  case ScalarTypeKind::SignedLong:
    BuiltinStr = "Li";
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
      BuiltinStr += "x";
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
  // Pointer to vector types. Defined for segment load intrinsics.
  // segment load intrinsics have pointer type arguments to store the loaded
  // vector values.
  if (IsPointer)
    BuiltinStr += "*";
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
    if (IsPointer)
      Str += " *";
    return;
  case ScalarTypeKind::Ptrdiff_t:
    Str = "ptrdiff_t";
    return;
  case ScalarTypeKind::UnsignedLong:
    Str = "unsigned long";
    return;
  case ScalarTypeKind::SignedLong:
    Str = "long";
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
      else if (ElementBitwidth == 16)
        Str += "_Float16";
      else
        llvm_unreachable("Unhandled floating type.");
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
  switch (ScalarType) {
  case ScalarTypeKind::Boolean:
    assert(isVector());
    ShortStr = "b" + utostr(64 / Scale.getValue());
    return;
  case ScalarTypeKind::Float:
    ShortStr = "f" + utostr(ElementBitwidth);
    break;
  case ScalarTypeKind::SignedInteger:
    ShortStr = "i" + utostr(ElementBitwidth);
    break;
  case ScalarTypeKind::UnsignedInteger:
    ShortStr = "u" + utostr(ElementBitwidth);
    break;
  default:
    PrintFatalError("Unhandled case!");
  }
  if (isVector())
    ShortStr += LMUL.str();
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
  case 'x':
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
  auto PType = Transformer.back();
  switch (PType) {
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
  case 'u':
    ScalarType = ScalarTypeKind::UnsignedLong;
    break;
  case 'l':
    ScalarType = ScalarTypeKind::SignedLong;
    break;
  default:
    PrintFatalError("Illegal primitive type transformers!");
  }
  Transformer = Transformer.drop_back();

  // Extract and compute complex type transformer. It can only appear one time.
  if (Transformer.startswith("(")) {
    size_t Idx = Transformer.find(')');
    assert(Idx != StringRef::npos);
    StringRef ComplexType = Transformer.slice(1, Idx);
    Transformer = Transformer.drop_front(Idx + 1);
    assert(!Transformer.contains('(') &&
           "Only allow one complex type transformer");

    auto UpdateAndCheckComplexProto = [&]() {
      Scale = LMUL.getScale(ElementBitwidth);
      const StringRef VectorPrototypes("vwqom");
      if (!VectorPrototypes.contains(PType))
        PrintFatalError("Complex type transformer only supports vector type!");
      if (Transformer.find_first_of("PCKWS") != StringRef::npos)
        PrintFatalError(
            "Illegal type transformer for Complex type transformer");
    };
    auto ComputeFixedLog2LMUL =
        [&](StringRef Value,
            std::function<bool(const int32_t &, const int32_t &)> Compare) {
          int32_t Log2LMUL;
          Value.getAsInteger(10, Log2LMUL);
          if (!Compare(Log2LMUL, LMUL.Log2LMUL)) {
            ScalarType = Invalid;
            return false;
          }
          // Update new LMUL
          LMUL = LMULType(Log2LMUL);
          UpdateAndCheckComplexProto();
          return true;
        };
    auto ComplexTT = ComplexType.split(":");
    if (ComplexTT.first == "Log2EEW") {
      uint32_t Log2EEW;
      ComplexTT.second.getAsInteger(10, Log2EEW);
      // update new elmul = (eew/sew) * lmul
      LMUL.MulLog2LMUL(Log2EEW - Log2_32(ElementBitwidth));
      // update new eew
      ElementBitwidth = 1 << Log2EEW;
      ScalarType = ScalarTypeKind::SignedInteger;
      UpdateAndCheckComplexProto();
    } else if (ComplexTT.first == "FixedSEW") {
      uint32_t NewSEW;
      ComplexTT.second.getAsInteger(10, NewSEW);
      // Set invalid type if src and dst SEW are same.
      if (ElementBitwidth == NewSEW) {
        ScalarType = Invalid;
        return;
      }
      // Update new SEW
      ElementBitwidth = NewSEW;
      UpdateAndCheckComplexProto();
    } else if (ComplexTT.first == "LFixedLog2LMUL") {
      // New LMUL should be larger than old
      if (!ComputeFixedLog2LMUL(ComplexTT.second, std::greater<int32_t>()))
        return;
    } else if (ComplexTT.first == "SFixedLog2LMUL") {
      // New LMUL should be smaller than old
      if (!ComputeFixedLog2LMUL(ComplexTT.second, std::less<int32_t>()))
        return;
    } else {
      PrintFatalError("Illegal complex type transformers!");
    }
  }

  // Compute the remain type transformers
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
RVVIntrinsic::RVVIntrinsic(
    StringRef NewName, StringRef Suffix, StringRef NewMangledName,
    StringRef MangledSuffix, StringRef IRName, bool IsMasked,
    bool HasMaskedOffOperand, bool HasVL, PolicyScheme Scheme,
    bool HasUnMaskedOverloaded, bool HasBuiltinAlias, StringRef ManualCodegen,
    const RVVTypes &OutInTypes, const std::vector<int64_t> &NewIntrinsicTypes,
    const std::vector<StringRef> &RequiredFeatures, unsigned NF)
    : IRName(IRName), IsMasked(IsMasked), HasVL(HasVL), Scheme(Scheme),
      HasUnMaskedOverloaded(HasUnMaskedOverloaded),
      HasBuiltinAlias(HasBuiltinAlias), ManualCodegen(ManualCodegen.str()),
      NF(NF) {

  // Init BuiltinName, Name and MangledName
  BuiltinName = NewName.str();
  Name = BuiltinName;
  if (NewMangledName.empty())
    MangledName = NewName.split("_").first.str();
  else
    MangledName = NewMangledName.str();
  if (!Suffix.empty())
    Name += "_" + Suffix.str();
  if (!MangledSuffix.empty())
    MangledName += "_" + MangledSuffix.str();
  if (IsMasked) {
    BuiltinName += "_m";
    Name += "_m";
  }

  // Init RISC-V extensions
  for (const auto &T : OutInTypes) {
    if (T->isFloatVector(16) || T->isFloat(16))
      RISCVPredefinedMacros |= RISCVPredefinedMacro::Zvfh;
    if (T->isFloatVector(32))
      RISCVPredefinedMacros |= RISCVPredefinedMacro::VectorMaxELenFp32;
    if (T->isFloatVector(64))
      RISCVPredefinedMacros |= RISCVPredefinedMacro::VectorMaxELenFp64;
    if (T->isVector(64))
      RISCVPredefinedMacros |= RISCVPredefinedMacro::VectorMaxELen64;
  }
  for (auto Feature : RequiredFeatures) {
    if (Feature == "RV64")
      RISCVPredefinedMacros |= RISCVPredefinedMacro::RV64;
    // Note: Full multiply instruction (mulh, mulhu, mulhsu, smul) for EEW=64
    // require V.
    if (Feature == "FullMultiply" &&
        (RISCVPredefinedMacros & RISCVPredefinedMacro::VectorMaxELen64))
      RISCVPredefinedMacros |= RISCVPredefinedMacro::V;
  }

  // Init OutputType and InputTypes
  OutputType = OutInTypes[0];
  InputTypes.assign(OutInTypes.begin() + 1, OutInTypes.end());

  // IntrinsicTypes is unmasked TA version index. Need to update it
  // if there is merge operand (It is always in first operand).
  IntrinsicTypes = NewIntrinsicTypes;
  if ((IsMasked && HasMaskedOffOperand) ||
      (!IsMasked && hasPassthruOperand())) {
    for (auto &I : IntrinsicTypes) {
      if (I >= 0)
        I += NF;
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
  if (!getIRName().empty())
    OS << "  ID = Intrinsic::riscv_" + getIRName() + ";\n";
  if (NF >= 2)
    OS << "  NF = " + utostr(getNF()) + ";\n";
  if (hasManualCodegen()) {
    OS << ManualCodegen;
    OS << "break;\n";
    return;
  }

  if (isMasked()) {
    if (hasVL()) {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);\n";
      if (hasPolicyOperand())
        OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(),"
              " TAIL_UNDISTURBED));\n";
    } else {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());\n";
    }
  } else {
    if (hasPolicyOperand())
      OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(), "
            "TAIL_UNDISTURBED));\n";
    else if (hasPassthruOperand()) {
      OS << "  Ops.push_back(llvm::UndefValue::get(ResultType));\n";
      OS << "  std::rotate(Ops.rbegin(), Ops.rbegin() + 1,  Ops.rend());\n";
    }
  }

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
    OS << ", Ops.back()->getType()";
  OS << "};\n";
  OS << "  break;\n";
}

void RVVIntrinsic::emitIntrinsicFuncDef(raw_ostream &OS) const {
  OS << "__attribute__((__clang_builtin_alias__(";
  OS << "__builtin_rvv_" << getBuiltinName() << ")))\n";
  OS << OutputType->getTypeStr() << " " << getName() << "(";
  // Emit function arguments
  if (!InputTypes.empty()) {
    ListSeparator LS;
    for (unsigned i = 0; i < InputTypes.size(); ++i)
      OS << LS << InputTypes[i]->getTypeStr();
  }
  OS << ");\n";
}

void RVVIntrinsic::emitMangledFuncDef(raw_ostream &OS) const {
  OS << "__attribute__((__clang_builtin_alias__(";
  OS << "__builtin_rvv_" << getBuiltinName() << ")))\n";
  OS << OutputType->getTypeStr() << " " << getMangledName() << "(";
  // Emit function arguments
  if (!InputTypes.empty()) {
    ListSeparator LS;
    for (unsigned i = 0; i < InputTypes.size(); ++i)
      OS << LS << InputTypes[i]->getTypeStr();
  }
  OS << ");\n";
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

  printHeaderCode(OS);

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
  OS << "#if defined(__riscv_zvfh)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = computeType('x', Log2LMUL, "v");
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
  for (int Log2LMUL : Log2LMULs) {
    auto T = computeType('d', Log2LMUL, "v");
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n\n";

  // The same extension include in the same arch guard marco.
  llvm::stable_sort(Defs, [](const std::unique_ptr<RVVIntrinsic> &A,
                             const std::unique_ptr<RVVIntrinsic> &B) {
    return A->getRISCVPredefinedMacros() < B->getRISCVPredefinedMacros();
  });

  OS << "#define __rvv_ai static __inline__\n";

  // Print intrinsic functions with macro
  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    OS << "__rvv_ai ";
    Inst.emitIntrinsicFuncDef(OS);
  });

  OS << "#undef __rvv_ai\n\n";

  OS << "#define __riscv_v_intrinsic_overloading 1\n";

  // Print Overloaded APIs
  OS << "#define __rvv_aio static __inline__ "
        "__attribute__((__overloadable__))\n";

  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    if (!Inst.isMasked() && !Inst.hasUnMaskedOverloaded())
      return;
    OS << "__rvv_aio ";
    Inst.emitMangledFuncDef(OS);
  });

  OS << "#undef __rvv_aio\n";

  OS << "\n#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif // __cplusplus\n";
  OS << "#endif // __RISCV_VECTOR_H\n";
}

void RVVEmitter::createBuiltins(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  OS << "#if defined(TARGET_BUILTIN) && !defined(RISCVV_BUILTIN)\n";
  OS << "#define RISCVV_BUILTIN(ID, TYPE, ATTRS) TARGET_BUILTIN(ID, TYPE, "
        "ATTRS, \"zve32x\")\n";
  OS << "#endif\n";
  for (auto &Def : Defs) {
    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (!P.second) {
      // Verf that this would have produced the same builtin definition.
      if (P.first->second->hasBuiltinAlias() != Def->hasBuiltinAlias())
        PrintFatalError("Builtin with same name has different hasAutoDef");
      else if (!Def->hasBuiltinAlias() &&
               P.first->second->getBuiltinTypeStr() != Def->getBuiltinTypeStr())
        PrintFatalError("Builtin with same name has different type string");
      continue;
    }
    OS << "RISCVV_BUILTIN(__builtin_rvv_" << Def->getBuiltinName() << ",\"";
    if (!Def->hasBuiltinAlias())
      OS << Def->getBuiltinTypeStr();
    OS << "\", \"n\")\n";
  }
  OS << "#undef RISCVV_BUILTIN\n";
}

void RVVEmitter::createCodeGen(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);
  // IR name could be empty, use the stable sort preserves the relative order.
  llvm::stable_sort(Defs, [](const std::unique_ptr<RVVIntrinsic> &A,
                             const std::unique_ptr<RVVIntrinsic> &B) {
    return A->getIRName() < B->getIRName();
  });

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  // Print switch body when the ir name or ManualCodegen changes from previous
  // iteration.
  RVVIntrinsic *PrevDef = Defs.begin()->get();
  for (auto &Def : Defs) {
    StringRef CurIRName = Def->getIRName();
    if (CurIRName != PrevDef->getIRName() ||
        (Def->getManualCodegen() != PrevDef->getManualCodegen())) {
      PrevDef->emitCodeGenSwitchBody(OS);
    }
    PrevDef = Def.get();

    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (P.second) {
      OS << "case RISCVVector::BI__builtin_rvv_" << Def->getBuiltinName()
         << ":\n";
      continue;
    }

    if (P.first->second->getIRName() != Def->getIRName())
      PrintFatalError("Builtin with same name has different IRName");
    else if (P.first->second->getManualCodegen() != Def->getManualCodegen())
      PrintFatalError("Builtin with same name has different ManualCodegen");
    else if (P.first->second->getNF() != Def->getNF())
      PrintFatalError("Builtin with same name has different NF");
    else if (P.first->second->isMasked() != Def->isMasked())
      PrintFatalError("Builtin with same name has different isMasked");
    else if (P.first->second->hasVL() != Def->hasVL())
      PrintFatalError("Builtin with same name has different hasVL");
    else if (P.first->second->getPolicyScheme() != Def->getPolicyScheme())
      PrintFatalError("Builtin with same name has different getPolicyScheme");
    else if (P.first->second->getIntrinsicTypes() != Def->getIntrinsicTypes())
      PrintFatalError("Builtin with same name has different IntrinsicTypes");
  }
  Defs.back()->emitCodeGenSwitchBody(OS);
  OS << "\n";
}

void RVVEmitter::parsePrototypes(StringRef Prototypes,
                                 std::function<void(StringRef)> Handler) {
  const StringRef Primaries("evwqom0ztul");
  while (!Prototypes.empty()) {
    size_t Idx = 0;
    // Skip over complex prototype because it could contain primitive type
    // character.
    if (Prototypes[0] == '(')
      Idx = Prototypes.find_first_of(')');
    Idx = Prototypes.find_first_of(Primaries, Idx);
    assert(Idx != StringRef::npos);
    Handler(Prototypes.slice(0, Idx + 1));
    Prototypes = Prototypes.drop_front(Idx + 1);
  }
}

std::string RVVEmitter::getSuffixStr(char Type, int Log2LMUL,
                                     StringRef Prototypes) {
  SmallVector<std::string> SuffixStrs;
  parsePrototypes(Prototypes, [&](StringRef Proto) {
    auto T = computeType(Type, Log2LMUL, Proto);
    SuffixStrs.push_back(T.getValue()->getShortStr());
  });
  return join(SuffixStrs, "_");
}

void RVVEmitter::createRVVIntrinsics(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Out) {
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("RVVBuiltin");
  for (auto *R : RV) {
    StringRef Name = R->getValueAsString("Name");
    StringRef SuffixProto = R->getValueAsString("Suffix");
    StringRef MangledName = R->getValueAsString("MangledName");
    StringRef MangledSuffixProto = R->getValueAsString("MangledSuffix");
    StringRef Prototypes = R->getValueAsString("Prototype");
    StringRef TypeRange = R->getValueAsString("TypeRange");
    bool HasMasked = R->getValueAsBit("HasMasked");
    bool HasMaskedOffOperand = R->getValueAsBit("HasMaskedOffOperand");
    bool HasVL = R->getValueAsBit("HasVL");
    Record *MaskedPolicyRecord = R->getValueAsDef("MaskedPolicy");
    PolicyScheme MaskedPolicy =
        static_cast<PolicyScheme>(MaskedPolicyRecord->getValueAsInt("Value"));
    Record *UnMaskedPolicyRecord = R->getValueAsDef("UnMaskedPolicy");
    PolicyScheme UnMaskedPolicy =
        static_cast<PolicyScheme>(UnMaskedPolicyRecord->getValueAsInt("Value"));
    bool HasUnMaskedOverloaded = R->getValueAsBit("HasUnMaskedOverloaded");
    std::vector<int64_t> Log2LMULList = R->getValueAsListOfInts("Log2LMUL");
    bool HasBuiltinAlias = R->getValueAsBit("HasBuiltinAlias");
    StringRef ManualCodegen = R->getValueAsString("ManualCodegen");
    StringRef MaskedManualCodegen = R->getValueAsString("MaskedManualCodegen");
    std::vector<int64_t> IntrinsicTypes =
        R->getValueAsListOfInts("IntrinsicTypes");
    std::vector<StringRef> RequiredFeatures =
        R->getValueAsListOfStrings("RequiredFeatures");
    StringRef IRName = R->getValueAsString("IRName");
    StringRef MaskedIRName = R->getValueAsString("MaskedIRName");
    unsigned NF = R->getValueAsInt("NF");

    // Parse prototype and create a list of primitive type with transformers
    // (operand) in ProtoSeq. ProtoSeq[0] is output operand.
    SmallVector<std::string> ProtoSeq;
    parsePrototypes(Prototypes, [&ProtoSeq](StringRef Proto) {
      ProtoSeq.push_back(Proto.str());
    });

    // Compute Builtin types
    SmallVector<std::string> ProtoMaskSeq = ProtoSeq;
    if (HasMasked) {
      // If HasMaskedOffOperand, insert result type as first input operand.
      if (HasMaskedOffOperand) {
        if (NF == 1) {
          ProtoMaskSeq.insert(ProtoMaskSeq.begin() + 1, ProtoSeq[0]);
        } else {
          // Convert
          // (void, op0 address, op1 address, ...)
          // to
          // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
          for (unsigned I = 0; I < NF; ++I)
            ProtoMaskSeq.insert(
                ProtoMaskSeq.begin() + NF + 1,
                ProtoSeq[1].substr(1)); // Use substr(1) to skip '*'
        }
      }
      if (HasMaskedOffOperand && NF > 1) {
        // Convert
        // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
        // to
        // (void, op0 address, op1 address, ..., mask, maskedoff0, maskedoff1,
        // ...)
        ProtoMaskSeq.insert(ProtoMaskSeq.begin() + NF + 1, "m");
      } else {
        // If HasMasked, insert 'm' as first input operand.
        ProtoMaskSeq.insert(ProtoMaskSeq.begin() + 1, "m");
      }
    }
    // If HasVL, append 'z' to last operand
    if (HasVL) {
      ProtoSeq.push_back("z");
      ProtoMaskSeq.push_back("z");
    }

    // Create Intrinsics for each type and LMUL.
    for (char I : TypeRange) {
      for (int Log2LMUL : Log2LMULList) {
        Optional<RVVTypes> Types = computeTypes(I, Log2LMUL, NF, ProtoSeq);
        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue())
          continue;

        auto SuffixStr = getSuffixStr(I, Log2LMUL, SuffixProto);
        auto MangledSuffixStr = getSuffixStr(I, Log2LMUL, MangledSuffixProto);
        // Create a unmasked intrinsic
        Out.push_back(std::make_unique<RVVIntrinsic>(
            Name, SuffixStr, MangledName, MangledSuffixStr, IRName,
            /*IsMasked=*/false, /*HasMaskedOffOperand=*/false, HasVL,
            UnMaskedPolicy, HasUnMaskedOverloaded, HasBuiltinAlias,
            ManualCodegen, Types.getValue(), IntrinsicTypes, RequiredFeatures,
            NF));
        if (HasMasked) {
          // Create a masked intrinsic
          Optional<RVVTypes> MaskTypes =
              computeTypes(I, Log2LMUL, NF, ProtoMaskSeq);
          Out.push_back(std::make_unique<RVVIntrinsic>(
              Name, SuffixStr, MangledName, MangledSuffixStr, MaskedIRName,
              /*IsMasked=*/true, HasMaskedOffOperand, HasVL, MaskedPolicy,
              HasUnMaskedOverloaded, HasBuiltinAlias, MaskedManualCodegen,
              MaskTypes.getValue(), IntrinsicTypes, RequiredFeatures, NF));
        }
      } // end for Log2LMULList
    }   // end for TypeRange
  }
}

void RVVEmitter::printHeaderCode(raw_ostream &OS) {
  std::vector<Record *> RVVHeaders =
      Records.getAllDerivedDefinitions("RVVHeader");
  for (auto *R : RVVHeaders) {
    StringRef HeaderCodeStr = R->getValueAsString("HeaderCode");
    OS << HeaderCodeStr.str();
  }
}

Optional<RVVTypes>
RVVEmitter::computeTypes(BasicType BT, int Log2LMUL, unsigned NF,
                         ArrayRef<std::string> PrototypeSeq) {
  // LMUL x NF must be less than or equal to 8.
  if ((Log2LMUL >= 1) && (1 << Log2LMUL) * NF > 8)
    return llvm::None;

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
  RISCVPredefinedMacroT PrevMacros =
      (*Defs.begin())->getRISCVPredefinedMacros();
  bool NeedEndif = emitMacroRestrictionStr(PrevMacros, OS);
  for (auto &Def : Defs) {
    RISCVPredefinedMacroT CurMacros = Def->getRISCVPredefinedMacros();
    if (CurMacros != PrevMacros) {
      if (NeedEndif)
        OS << "#endif\n\n";
      NeedEndif = emitMacroRestrictionStr(CurMacros, OS);
      PrevMacros = CurMacros;
    }
    if (Def->hasBuiltinAlias())
      PrintBody(OS, *Def);
  }
  if (NeedEndif)
    OS << "#endif\n\n";
}

bool RVVEmitter::emitMacroRestrictionStr(RISCVPredefinedMacroT PredefinedMacros,
                                         raw_ostream &OS) {
  if (PredefinedMacros == RISCVPredefinedMacro::Basic)
    return false;
  OS << "#if ";
  ListSeparator LS(" && ");
  if (PredefinedMacros & RISCVPredefinedMacro::V)
    OS << LS << "defined(__riscv_v)";
  if (PredefinedMacros & RISCVPredefinedMacro::Zvfh)
    OS << LS << "defined(__riscv_zvfh)";
  if (PredefinedMacros & RISCVPredefinedMacro::RV64)
    OS << LS << "(__riscv_xlen == 64)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELen64)
    OS << LS << "(__riscv_v_elen >= 64)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELenFp32)
    OS << LS << "(__riscv_v_elen_fp >= 32)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELenFp64)
    OS << LS << "(__riscv_v_elen_fp >= 64)";
  OS << "\n";
  return true;
}

namespace clang {
void EmitRVVHeader(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createHeader(OS);
}

void EmitRVVBuiltins(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createBuiltins(OS);
}

void EmitRVVBuiltinCG(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createCodeGen(OS);
}

} // End namespace clang
