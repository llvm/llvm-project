//===- RISCVVIntrinsicUtils.cpp - RISC-V Vector Intrinsic Utils -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace llvm;

namespace clang {
namespace RISCV {

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
    llvm_unreachable("Unhandled case!");
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
    llvm_unreachable("Unhandled type code!");
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
    llvm_unreachable("Illegal primitive type transformers!");
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
        llvm_unreachable("Complex type transformer only supports vector type!");
      if (Transformer.find_first_of("PCKWS") != StringRef::npos)
        llvm_unreachable(
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
      llvm_unreachable("Illegal complex type transformers!");
    }
  }

  // Compute the remain type transformers
  for (char I : Transformer) {
    switch (I) {
    case 'P':
      if (IsConstant)
        llvm_unreachable("'P' transformer cannot be used after 'C'");
      if (IsPointer)
        llvm_unreachable("'P' transformer cannot be used twice");
      IsPointer = true;
      break;
    case 'C':
      if (IsConstant)
        llvm_unreachable("'C' transformer cannot be used twice");
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
      llvm_unreachable("Illegal non-primitive type transformer!");
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

} // end namespace RISCV
} // end namespace clang
