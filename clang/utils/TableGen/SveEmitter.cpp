//===- SveEmitter.cpp - Generate arm_sve.h for use with clang -*- C++ -*-===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_sve.h, which includes
// a declaration and definition of each function specified by the ARM C/C++
// Language Extensions (ACLE).
//
// For details, visit:
//  https://developer.arm.com/architectures/system-architectures/software-standards/acle
//
// Each SVE instruction is implemented in terms of 1 or more functions which
// are suffixed with the element type of the input vectors.  Functions may be
// implemented in terms of generic vector operations such as +, *, -, etc. or
// by calling a __builtin_-prefixed function which will be handled by clang's
// CodeGen library.
//
// See also the documentation in include/clang/Basic/arm_sve.td.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/Error.h"
#include <string>
#include <sstream>
#include <set>
#include <cctype>

using namespace llvm;

enum ClassKind {
  ClassNone,
  ClassS,     // signed/unsigned, e.g., "_s8", "_u8" suffix
  ClassG,     // Overloaded name without type suffix
};

using TypeSpec = std::string;

namespace {

class SVEType {
  TypeSpec TS;
  bool Float, Signed, Immediate, Void, Constant, Pointer;
  bool DefaultType, IsScalable, Predicate, PredicatePattern, PrefetchOp;
  unsigned Bitwidth, ElementBitwidth, NumVectors;

public:
  SVEType() : SVEType(TypeSpec(), 'v') {}

  SVEType(TypeSpec TS, char CharMod)
      : TS(TS), Float(false), Signed(true), Immediate(false), Void(false),
        Constant(false), Pointer(false), DefaultType(false), IsScalable(true),
        Predicate(false), PredicatePattern(false), PrefetchOp(false),
        Bitwidth(128), ElementBitwidth(~0U), NumVectors(1) {
    if (!TS.empty())
      applyTypespec();
    applyModifier(CharMod);
  }

  /// Return the value in SVETypeFlags for this type.
  unsigned getTypeFlags() const;

  bool isPointer() const { return Pointer; }
  bool isVoidPointer() const { return Pointer && Void; }
  bool isSigned() const { return Signed; }
  bool isImmediate() const { return Immediate; }
  bool isScalar() const { return NumVectors == 0; }
  bool isVector() const { return NumVectors > 0; }
  bool isScalableVector() const { return isVector() && IsScalable; }
  bool isChar() const { return ElementBitwidth == 8; }
  bool isVoid() const { return Void & !Pointer; }
  bool isDefault() const { return DefaultType; }
  bool isFloat() const { return Float; }
  bool isInteger() const { return !Float && !Predicate; }
  bool isScalarPredicate() const { return !Float && ElementBitwidth == 1; }
  bool isPredicateVector() const { return Predicate; }
  bool isPredicatePattern() const { return PredicatePattern; }
  bool isPrefetchOp() const { return PrefetchOp; }
  bool isConstant() const { return Constant; }
  unsigned getElementSizeInBits() const { return ElementBitwidth; }
  unsigned getNumVectors() const { return NumVectors; }

  unsigned getNumElements() const {
    assert(ElementBitwidth != ~0U);
    return Bitwidth / ElementBitwidth;
  }
  unsigned getSizeInBits() const {
    return Bitwidth;
  }

  /// Return the string representation of a type, which is an encoded
  /// string for passing to the BUILTIN() macro in Builtins.def.
  std::string builtin_str() const;

  /// Return the C/C++ string representation of a type for use in the
  /// arm_sve.h header file.
  std::string str() const;

private:
  /// Creates the type based on the typespec string in TS.
  void applyTypespec();

  /// Applies a prototype modifier to the type.
  void applyModifier(char Mod);
};


class SVEEmitter;

/// The main grunt class. This represents an instantiation of an intrinsic with
/// a particular typespec and prototype.
class Intrinsic {
  /// The unmangled name.
  std::string Name;

  /// The name of the corresponding LLVM IR intrinsic.
  std::string LLVMName;

  /// Intrinsic prototype.
  std::string Proto;

  /// The base type spec for this intrinsic.
  TypeSpec BaseTypeSpec;

  /// The base class kind. Most intrinsics use ClassS, which has full type
  /// info for integers (_s32/_u32), or ClassG which is used for overloaded
  /// intrinsics.
  ClassKind Class;

  /// The architectural #ifdef guard.
  std::string Guard;

  /// The types of return value [0] and parameters [1..].
  std::vector<SVEType> Types;

  /// The "base type", which is VarType('d', BaseTypeSpec).
  SVEType BaseType;

  unsigned Flags;

public:
  /// The type of predication.
  enum MergeType {
    MergeNone,
    MergeAny,
    MergeOp1,
    MergeZero,
    MergeAnyExp,
    MergeZeroExp,
    MergeInvalid
  } Merge;

  Intrinsic(StringRef Name, StringRef Proto, int64_t MT, StringRef LLVMName,
            unsigned Flags, TypeSpec BT, ClassKind Class, SVEEmitter &Emitter,
            StringRef Guard)
      : Name(Name.str()), LLVMName(LLVMName), Proto(Proto.str()),
        BaseTypeSpec(BT), Class(Class), Guard(Guard.str()), BaseType(BT, 'd'),
        Flags(Flags), Merge(MergeType(MT)) {
    // Types[0] is the return value.
    for (unsigned I = 0; I < Proto.size(); ++I)
      Types.emplace_back(BaseTypeSpec, Proto[I]);
  }

  ~Intrinsic()=default;

  std::string getName() const { return Name; }
  std::string getLLVMName() const { return LLVMName; }
  std::string getProto() const { return Proto; }
  TypeSpec getBaseTypeSpec() const { return BaseTypeSpec; }
  SVEType getBaseType() const { return BaseType; }

  StringRef getGuard() const { return Guard; }
  ClassKind getClassKind() const { return Class; }
  MergeType getMergeType() const { return Merge; }

  SVEType getReturnType() const { return Types[0]; }
  ArrayRef<SVEType> getTypes() const { return Types; }
  SVEType getParamType(unsigned I) const { return Types[I + 1]; }
  unsigned getNumParams() const { return Proto.size() - 1; }

  unsigned getFlags() const { return Flags; }
  bool isFlagSet(uint64_t Flag) const { return Flags & Flag;}

  /// Return the type string for a BUILTIN() macro in Builtins.def.
  std::string getBuiltinTypeStr();

  /// Return the name, mangled with type information. The name is mangled for
  /// ClassS, so will add type suffixes such as _u32/_s32.
  std::string getMangledName() const { return mangleName(ClassS); }

  /// Returns true if the intrinsic is overloaded, in that it should also generate
  /// a short form without the type-specifiers, e.g. 'svld1(..)' instead of
  /// 'svld1_u32(..)'.
  static bool isOverloadedIntrinsic(StringRef Name) {
    auto BrOpen = Name.find("[");
    auto BrClose = Name.find(']');
    return BrOpen != std::string::npos && BrClose != std::string::npos;
  }

  /// Emits the intrinsic declaration to the ostream.
  void emitIntrinsic(raw_ostream &OS) const;

private:
  std::string getMergeSuffix() const;
  std::string mangleName(ClassKind LocalCK) const;
  std::string replaceTemplatedArgs(std::string Name, TypeSpec TS,
                                   std::string Proto) const;
};

class SVEEmitter {
private:
  RecordKeeper &Records;
  llvm::StringMap<uint64_t> EltTypes;
  llvm::StringMap<uint64_t> MemEltTypes;
  llvm::StringMap<uint64_t> FlagTypes;

  unsigned getTypeFlags(const SVEType &T);
public:
  SVEEmitter(RecordKeeper &R) : Records(R) {
    for (auto *RV : Records.getAllDerivedDefinitions("EltType"))
      EltTypes[RV->getNameInitAsString()] = RV->getValueAsInt("Value");
    for (auto *RV : Records.getAllDerivedDefinitions("MemEltType"))
      MemEltTypes[RV->getNameInitAsString()] = RV->getValueAsInt("Value");
    for (auto *RV : Records.getAllDerivedDefinitions("FlagType"))
      FlagTypes[RV->getNameInitAsString()] = RV->getValueAsInt("Value");
  }

  /// Emit arm_sve.h.
  void createHeader(raw_ostream &o);

  /// Emit all the __builtin prototypes and code needed by Sema.
  void createBuiltins(raw_ostream &o);

  /// Emit all the information needed to map builtin -> LLVM IR intrinsic.
  void createCodeGenMap(raw_ostream &o);

  /// Create the SVETypeFlags used in CGBuiltins
  void createTypeFlags(raw_ostream &o);

  /// Create intrinsic and add it to \p Out
  void createIntrinsic(Record *R, SmallVectorImpl<std::unique_ptr<Intrinsic>> &Out);
};

} // end anonymous namespace


//===----------------------------------------------------------------------===//
// Type implementation
//===----------------------------------------------------------------------===//

unsigned SVEEmitter::getTypeFlags(const SVEType &T) {
  unsigned FirstEltType = EltTypes["FirstEltType"];
  if (T.isFloat()) {
    switch (T.getElementSizeInBits()) {
    case 16: return FirstEltType + EltTypes["EltTyFloat16"];
    case 32: return FirstEltType + EltTypes["EltTyFloat32"];
    case 64: return FirstEltType + EltTypes["EltTyFloat64"];
    default: llvm_unreachable("Unhandled float element bitwidth!");
    }
  }

  if (T.isPredicateVector()) {
    switch (T.getElementSizeInBits()) {
    case 8:  return FirstEltType + EltTypes["EltTyBool8"];
    case 16: return FirstEltType + EltTypes["EltTyBool16"];
    case 32: return FirstEltType + EltTypes["EltTyBool32"];
    case 64: return FirstEltType + EltTypes["EltTyBool64"];
    default: llvm_unreachable("Unhandled predicate element bitwidth!");
    }
  }

  switch (T.getElementSizeInBits()) {
  case 8:  return FirstEltType + EltTypes["EltTyInt8"];
  case 16: return FirstEltType + EltTypes["EltTyInt16"];
  case 32: return FirstEltType + EltTypes["EltTyInt32"];
  case 64: return FirstEltType + EltTypes["EltTyInt64"];
  default: llvm_unreachable("Unhandled integer element bitwidth!");
  }
}

std::string SVEType::builtin_str() const {
  std::string S;
  if (isVoid())
    return "v";

  if (isVoidPointer())
    S += "v";
  else if (!Float)
    switch (ElementBitwidth) {
    case 1: S += "b"; break;
    case 8: S += "c"; break;
    case 16: S += "s"; break;
    case 32: S += "i"; break;
    case 64: S += "Wi"; break;
    case 128: S += "LLLi"; break;
    default: llvm_unreachable("Unhandled case!");
    }
  else
    switch (ElementBitwidth) {
    case 16: S += "h"; break;
    case 32: S += "f"; break;
    case 64: S += "d"; break;
    default: llvm_unreachable("Unhandled case!");
    }

  if (!isFloat()) {
    if ((isChar() || isPointer()) && !isVoidPointer()) {
      // Make chars and typed pointers explicitly signed.
      if (Signed)
        S = "S" + S;
      else if (!Signed)
        S = "U" + S;
    } else if (!isVoidPointer() && !Signed) {
      S = "U" + S;
    }
  }

  // Constant indices are "int", but have the "constant expression" modifier.
  if (isImmediate()) {
    assert(!isFloat() && "fp immediates are not supported");
    S = "I" + S;
  }

  if (isScalar()) {
    if (Constant) S += "C";
    if (Pointer) S += "*";
    return S;
  }

  assert(isScalableVector() && "Unsupported type");
  return "q" + utostr(getNumElements() * NumVectors) + S;
}

std::string SVEType::str() const {
  if (isPredicatePattern())
    return "sv_pattern";

  if (isPrefetchOp())
    return "sv_prfop";

  std::string S;
  if (Void)
    S += "void";
  else {
    if (isScalableVector())
      S += "sv";
    if (!Signed && !Float)
      S += "u";

    if (Float)
      S += "float";
    else if (isScalarPredicate())
      S += "bool";
    else
      S += "int";

    if (!isScalarPredicate())
      S += utostr(ElementBitwidth);
    if (!isScalableVector() && isVector())
      S += "x" + utostr(getNumElements());
    if (NumVectors > 1)
      S += "x" + utostr(NumVectors);
    S += "_t";
  }

  if (Constant)
    S += " const";
  if (Pointer)
    S += " *";

  return S;
}
void SVEType::applyTypespec() {
  for (char I : TS) {
    switch (I) {
    case 'P':
      Predicate = true;
      ElementBitwidth = 1;
      break;
    case 'U':
      Signed = false;
      break;
    case 'c':
      ElementBitwidth = 8;
      break;
    case 's':
      ElementBitwidth = 16;
      break;
    case 'i':
      ElementBitwidth = 32;
      break;
    case 'l':
      ElementBitwidth = 64;
      break;
    case 'h':
      Float = true;
      ElementBitwidth = 16;
      break;
    case 'f':
      Float = true;
      ElementBitwidth = 32;
      break;
    case 'd':
      Float = true;
      ElementBitwidth = 64;
      break;
    default:
      llvm_unreachable("Unhandled type code!");
    }
  }
  assert(ElementBitwidth != ~0U && "Bad element bitwidth!");
}

void SVEType::applyModifier(char Mod) {
  switch (Mod) {
  case 'v':
    Void = true;
    break;
  case 'd':
    DefaultType = true;
    break;
  case 'c':
    Constant = true;
    LLVM_FALLTHROUGH;
  case 'p':
    Pointer = true;
    Bitwidth = ElementBitwidth;
    NumVectors = 0;
    break;
  case 'P':
    Signed = true;
    Float = false;
    Predicate = true;
    Bitwidth = 16;
    ElementBitwidth = 1;
    break;
  default:
    llvm_unreachable("Unhandled character!");
  }
}


//===----------------------------------------------------------------------===//
// Intrinsic implementation
//===----------------------------------------------------------------------===//

std::string Intrinsic::getBuiltinTypeStr() {
  std::string S;

  SVEType RetT = getReturnType();
  // Since the return value must be one type, return a vector type of the
  // appropriate width which we will bitcast.  An exception is made for
  // returning structs of 2, 3, or 4 vectors which are returned in a sret-like
  // fashion, storing them to a pointer arg.
  if (RetT.getNumVectors() > 1) {
    S += "vv*"; // void result with void* first argument
  } else
    S += RetT.builtin_str();

  for (unsigned I = 0; I < getNumParams(); ++I)
    S += getParamType(I).builtin_str();

  return S;
}

std::string Intrinsic::replaceTemplatedArgs(std::string Name, TypeSpec TS,
                                            std::string Proto) const {
  std::string Ret = Name;
  while (Ret.find('{') != std::string::npos) {
    size_t Pos = Ret.find('{');
    size_t End = Ret.find('}');
    unsigned NumChars = End - Pos + 1;
    assert(NumChars == 3 && "Unexpected template argument");

    SVEType T;
    char C = Ret[Pos+1];
    switch(C) {
    default:
      llvm_unreachable("Unknown predication specifier");
    case 'd':
      T = SVEType(TS, 'd');
      break;
    case '0':
    case '1':
    case '2':
    case '3':
      T = SVEType(TS, Proto[C - '0']);
      break;
    }

    // Replace templated arg with the right suffix (e.g. u32)
    std::string TypeCode;
    if (T.isInteger())
      TypeCode = T.isSigned() ? 's' : 'u';
    else if (T.isPredicateVector())
      TypeCode = 'b';
    else
      TypeCode = 'f';
    Ret.replace(Pos, NumChars, TypeCode + utostr(T.getElementSizeInBits()));
  }

  return Ret;
}

// ACLE function names have a merge style postfix.
std::string Intrinsic::getMergeSuffix() const {
  switch (getMergeType()) {
    default:
      llvm_unreachable("Unknown predication specifier");
    case MergeNone:    return "";
    case MergeAny:
    case MergeAnyExp:  return "_x";
    case MergeOp1:     return "_m";
    case MergeZero:
    case MergeZeroExp: return "_z";
  }
}

std::string Intrinsic::mangleName(ClassKind LocalCK) const {
  std::string S = getName();

  if (LocalCK == ClassG) {
    // Remove the square brackets and everything in between.
    while (S.find("[") != std::string::npos) {
      auto Start = S.find("[");
      auto End = S.find(']');
      S.erase(Start, (End-Start)+1);
    }
  } else {
    // Remove the square brackets.
    while (S.find("[") != std::string::npos) {
      auto BrPos = S.find('[');
      if (BrPos != std::string::npos)
        S.erase(BrPos, 1);
      BrPos = S.find(']');
      if (BrPos != std::string::npos)
        S.erase(BrPos, 1);
    }
  }

  // Replace all {d} like expressions with e.g. 'u32'
  return replaceTemplatedArgs(S, getBaseTypeSpec(), getProto()) +
         getMergeSuffix();
}

void Intrinsic::emitIntrinsic(raw_ostream &OS) const {
  // Use the preprocessor to 
  if (getClassKind() != ClassG || getProto().size() <= 1) {
    OS << "#define " << mangleName(getClassKind())
       << "(...) __builtin_sve_" << mangleName(ClassS)
       << "(__VA_ARGS__)\n";
  } else {
    std::string FullName = mangleName(ClassS);
    std::string ProtoName = mangleName(ClassG);

    OS << "__aio __attribute__((__clang_arm_builtin_alias("
       << "__builtin_sve_" << FullName << ")))\n";

    OS << getTypes()[0].str() << " " << ProtoName << "(";
    for (unsigned I = 0; I < getTypes().size() - 1; ++I) {
      if (I != 0)
        OS << ", ";
      OS << getTypes()[I + 1].str();
    }
    OS << ");\n";
  }
}

//===----------------------------------------------------------------------===//
// SVEEmitter implementation
//===----------------------------------------------------------------------===//
void SVEEmitter::createIntrinsic(
    Record *R, SmallVectorImpl<std::unique_ptr<Intrinsic>> &Out) {
  StringRef Name = R->getValueAsString("Name");
  StringRef Proto = R->getValueAsString("Prototype");
  StringRef Types = R->getValueAsString("Types");
  StringRef Guard = R->getValueAsString("ArchGuard");
  StringRef LLVMName = R->getValueAsString("LLVMIntrinsic");
  int64_t Merge = R->getValueAsInt("Merge");
  std::vector<Record*> FlagsList = R->getValueAsListOfDefs("Flags");

  int64_t Flags = 0;
  for (auto FlagRec : FlagsList)
    Flags |= FlagRec->getValueAsInt("Value");
  Flags |= R->getValueAsInt("MemEltType") + MemEltTypes["FirstMemEltType"];

  // Extract type specs from string
  SmallVector<TypeSpec, 8> TypeSpecs;
  TypeSpec Acc;
  for (char I : Types) {
    Acc.push_back(I);
    if (islower(I)) {
      TypeSpecs.push_back(TypeSpec(Acc));
      Acc.clear();
    }
  }

  // Remove duplicate type specs.
  std::sort(TypeSpecs.begin(), TypeSpecs.end());
  TypeSpecs.erase(std::unique(TypeSpecs.begin(), TypeSpecs.end()),
                  TypeSpecs.end());

  // Create an Intrinsic for each type spec.
  for (auto TS : TypeSpecs) {
    Out.push_back(std::make_unique<Intrinsic>(Name, Proto, Merge,
                                              LLVMName, Flags, TS, ClassS,
                                              *this, Guard));

    // Also generate the short-form (e.g. svadd_m) for the given type-spec.
    if (Intrinsic::isOverloadedIntrinsic(Name))
      Out.push_back(std::make_unique<Intrinsic>(
          Name, Proto, Merge, LLVMName, Flags, TS, ClassG, *this, Guard));
  }
}

void SVEEmitter::createHeader(raw_ostream &OS) {
  OS << "/*===---- arm_sve.h - ARM SVE intrinsics "
        "-----------------------------------===\n"
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

  OS << "#ifndef __ARM_SVE_H\n";
  OS << "#define __ARM_SVE_H\n\n";

  OS << "#if !defined(__ARM_FEATURE_SVE)\n";
  OS << "#error \"SVE support not enabled\"\n";
  OS << "#else\n\n";

  OS << "#include <stdint.h>\n\n";
  OS << "#ifdef  __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#else\n";
  OS << "#include <stdbool.h>\n";
  OS << "#endif\n\n";

  OS << "typedef __fp16 float16_t;\n";
  OS << "typedef float float32_t;\n";
  OS << "typedef double float64_t;\n";
  OS << "typedef bool bool_t;\n\n";

  OS << "typedef __SVInt8_t svint8_t;\n";
  OS << "typedef __SVInt16_t svint16_t;\n";
  OS << "typedef __SVInt32_t svint32_t;\n";
  OS << "typedef __SVInt64_t svint64_t;\n";
  OS << "typedef __SVUint8_t svuint8_t;\n";
  OS << "typedef __SVUint16_t svuint16_t;\n";
  OS << "typedef __SVUint32_t svuint32_t;\n";
  OS << "typedef __SVUint64_t svuint64_t;\n";
  OS << "typedef __SVFloat16_t svfloat16_t;\n";
  OS << "typedef __SVFloat32_t svfloat32_t;\n";
  OS << "typedef __SVFloat64_t svfloat64_t;\n";
  OS << "typedef __SVBool_t  svbool_t;\n\n";

  OS << "/* Function attributes */\n";
  OS << "#define __aio static inline __attribute__((__always_inline__, "
        "__nodebug__, __overloadable__))\n\n";

  SmallVector<std::unique_ptr<Intrinsic>, 128> Defs;
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("Inst");
  for (auto *R : RV)
    createIntrinsic(R, Defs);

  // Sort intrinsics in header file by following order/priority:
  // - Architectural guard (i.e. does it require SVE2 or SVE2_AES)
  // - Class (is intrinsic overloaded or not)
  // - Intrinsic name
  std::stable_sort(
      Defs.begin(), Defs.end(), [](const std::unique_ptr<Intrinsic> &A,
                                   const std::unique_ptr<Intrinsic> &B) {
        return A->getGuard() < B->getGuard() ||
               (unsigned)A->getClassKind() < (unsigned)B->getClassKind() ||
               A->getName() < B->getName();
      });

  StringRef InGuard = "";
  for (auto &I : Defs) {
    // Emit #endif/#if pair if needed.
    if (I->getGuard() != InGuard) {
      if (!InGuard.empty())
        OS << "#endif  //" << InGuard << "\n";
      InGuard = I->getGuard();
      if (!InGuard.empty())
        OS << "\n#if " << InGuard << "\n";
    }

    // Actually emit the intrinsic declaration.
    I->emitIntrinsic(OS);
  }

  if (!InGuard.empty())
    OS << "#endif  //" << InGuard << "\n";

  OS << "#ifdef __cplusplus\n";
  OS << "} // extern \"C\"\n";
  OS << "#endif\n\n";
  OS << "#endif /*__ARM_FEATURE_SVE */\n\n";
  OS << "#endif /* __ARM_SVE_H */\n";
}

void SVEEmitter::createBuiltins(raw_ostream &OS) {
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("Inst");
  SmallVector<std::unique_ptr<Intrinsic>, 128> Defs;
  for (auto *R : RV)
    createIntrinsic(R, Defs);

  // The mappings must be sorted based on BuiltinID.
  llvm::sort(Defs, [](const std::unique_ptr<Intrinsic> &A,
                      const std::unique_ptr<Intrinsic> &B) {
    return A->getMangledName() < B->getMangledName();
  });

  OS << "#ifdef GET_SVE_BUILTINS\n";
  for (auto &Def : Defs) {
    // Only create BUILTINs for non-overloaded intrinsics, as overloaded
    // declarations only live in the header file.
    if (Def->getClassKind() != ClassG)
      OS << "BUILTIN(__builtin_sve_" << Def->getMangledName() << ", \""
         << Def->getBuiltinTypeStr() << "\", \"n\")\n";
  }
  OS << "#endif\n\n";
}

void SVEEmitter::createCodeGenMap(raw_ostream &OS) {
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("Inst");
  SmallVector<std::unique_ptr<Intrinsic>, 128> Defs;
  for (auto *R : RV)
    createIntrinsic(R, Defs);

  // The mappings must be sorted based on BuiltinID.
  llvm::sort(Defs, [](const std::unique_ptr<Intrinsic> &A,
                      const std::unique_ptr<Intrinsic> &B) {
    return A->getMangledName() < B->getMangledName();
  });

  OS << "#ifdef GET_SVE_LLVM_INTRINSIC_MAP\n";
  for (auto &Def : Defs) {
    // Builtins only exist for non-overloaded intrinsics, overloaded
    // declarations only live in the header file.
    if (Def->getClassKind() == ClassG)
      continue;

    uint64_t Flags = Def->getFlags() | getTypeFlags(Def->getBaseType());
    auto FlagString = std::to_string(Flags);

    std::string LLVMName = Def->getLLVMName();
    std::string Builtin = Def->getMangledName();
    if (!LLVMName.empty())
      OS << "SVEMAP1(" << Builtin << ", " << LLVMName << ", " << FlagString
         << "),\n";
    else
      OS << "SVEMAP2(" << Builtin << ", " << FlagString << "),\n";
  }
  OS << "#endif\n\n";
}

/// Create the SVETypeFlags used in CGBuiltins
void SVEEmitter::createTypeFlags(raw_ostream &OS) {
  OS << "#ifdef LLVM_GET_SVE_TYPEFLAGS\n";
  for (auto &KV : FlagTypes)
    OS << "const uint64_t " << KV.getKey() << " = " << KV.getValue() << ";\n";
  OS << "#endif\n\n";

  OS << "#ifdef LLVM_GET_SVE_ELTTYPES\n";
  for (auto &KV : EltTypes)
    OS << "  " << KV.getKey() << " = " << KV.getValue() << ",\n";
  OS << "#endif\n\n";

  OS << "#ifdef LLVM_GET_SVE_MEMELTTYPES\n";
  for (auto &KV : MemEltTypes)
    OS << "  " << KV.getKey() << " = " << KV.getValue() << ",\n";
  OS << "#endif\n\n";
}

namespace clang {
void EmitSveHeader(RecordKeeper &Records, raw_ostream &OS) {
  SVEEmitter(Records).createHeader(OS);
}

void EmitSveBuiltins(RecordKeeper &Records, raw_ostream &OS) {
  SVEEmitter(Records).createBuiltins(OS);
}

void EmitSveBuiltinCG(RecordKeeper &Records, raw_ostream &OS) {
  SVEEmitter(Records).createCodeGenMap(OS);
}
void EmitSveTypeFlags(RecordKeeper &Records, raw_ostream &OS) {
  SVEEmitter(Records).createTypeFlags(OS);
}

} // End namespace clang
