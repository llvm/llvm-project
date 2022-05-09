//===--- RISCVVIntrinsicUtils.h - RISC-V Vector Intrinsic Utils -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_SUPPORT_RISCVVINTRINSICUTILS_H
#define CLANG_SUPPORT_RISCVVINTRINSICUTILS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>

namespace clang {
namespace RISCV {

using BasicType = char;
using VScaleVal = llvm::Optional<unsigned>;

// Exponential LMUL
struct LMULType {
  int Log2LMUL;
  LMULType(int Log2LMUL);
  // Return the C/C++ string representation of LMUL
  std::string str() const;
  llvm::Optional<unsigned> getScale(unsigned ElementBitwidth) const;
  void MulLog2LMUL(int Log2LMUL);
  LMULType &operator*=(uint32_t RHS);
};

// This class is compact representation of a valid and invalid RVVType.
class RVVType {
  enum ScalarTypeKind : uint32_t {
    Void,
    Size_t,
    Ptrdiff_t,
    UnsignedLong,
    SignedLong,
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
  RVVType() : RVVType(BasicType(), 0, llvm::StringRef()) {}
  RVVType(BasicType BT, int Log2LMUL, llvm::StringRef prototype);

  // Return the string representation of a type, which is an encoded string for
  // passing to the BUILTIN() macro in Builtins.def.
  const std::string &getBuiltinStr() const { return BuiltinStr; }

  // Return the clang builtin type for RVV vector type which are used in the
  // riscv_vector.h header file.
  const std::string &getClangBuiltinStr() const { return ClangBuiltinStr; }

  // Return the C/C++ string representation of a type for use in the
  // riscv_vector.h header file.
  const std::string &getTypeStr() const { return Str; }

  // Return the short name of a type for C/C++ name suffix.
  const std::string &getShortStr() {
    // Not all types are used in short name, so compute the short name by
    // demanded.
    if (ShortStr.empty())
      initShortStr();
    return ShortStr;
  }

  bool isValid() const { return Valid; }
  bool isScalar() const { return Scale.hasValue() && Scale.getValue() == 0; }
  bool isVector() const { return Scale.hasValue() && Scale.getValue() != 0; }
  bool isVector(unsigned Width) const {
    return isVector() && ElementBitwidth == Width;
  }
  bool isFloat() const { return ScalarType == ScalarTypeKind::Float; }
  bool isSignedInteger() const {
    return ScalarType == ScalarTypeKind::SignedInteger;
  }
  bool isFloatVector(unsigned Width) const {
    return isVector() && isFloat() && ElementBitwidth == Width;
  }
  bool isFloat(unsigned Width) const {
    return isFloat() && ElementBitwidth == Width;
  }

private:
  // Verify RVV vector type and set Valid.
  bool verifyType() const;

  // Creates a type based on basic types of TypeRange
  void applyBasicType();

  // Applies a prototype modifier to the current type. The result maybe an
  // invalid type.
  void applyModifier(llvm::StringRef prototype);

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
using RISCVPredefinedMacroT = uint8_t;

enum RISCVPredefinedMacro : RISCVPredefinedMacroT {
  Basic = 0,
  V = 1 << 1,
  Zvfh = 1 << 2,
  RV64 = 1 << 3,
  VectorMaxELen64 = 1 << 4,
  VectorMaxELenFp32 = 1 << 5,
  VectorMaxELenFp64 = 1 << 6,
};

enum PolicyScheme : uint8_t {
  SchemeNone,
  HasPassthruOperand,
  HasPolicyOperand,
};

// TODO refactor RVVIntrinsic class design after support all intrinsic
// combination. This represents an instantiation of an intrinsic with a
// particular type and prototype
class RVVIntrinsic {

private:
  std::string BuiltinName; // Builtin name
  std::string Name;        // C intrinsic name.
  std::string MangledName;
  std::string IRName;
  bool IsMasked;
  bool HasVL;
  PolicyScheme Scheme;
  bool HasUnMaskedOverloaded;
  bool HasBuiltinAlias;
  std::string ManualCodegen;
  RVVTypePtr OutputType; // Builtin output type
  RVVTypes InputTypes;   // Builtin input types
  // The types we use to obtain the specific LLVM intrinsic. They are index of
  // InputTypes. -1 means the return type.
  std::vector<int64_t> IntrinsicTypes;
  RISCVPredefinedMacroT RISCVPredefinedMacros = 0;
  unsigned NF = 1;

public:
  RVVIntrinsic(llvm::StringRef Name, llvm::StringRef Suffix, llvm::StringRef MangledName,
               llvm::StringRef MangledSuffix, llvm::StringRef IRName, bool IsMasked,
               bool HasMaskedOffOperand, bool HasVL, PolicyScheme Scheme,
               bool HasUnMaskedOverloaded, bool HasBuiltinAlias,
               llvm::StringRef ManualCodegen, const RVVTypes &Types,
               const std::vector<int64_t> &IntrinsicTypes,
               const std::vector<llvm::StringRef> &RequiredFeatures, unsigned NF);
  ~RVVIntrinsic() = default;

  RVVTypePtr getOutputType() const { return OutputType; }
  const RVVTypes &getInputTypes() const { return InputTypes; }
  llvm::StringRef getBuiltinName() const { return BuiltinName; }
  llvm::StringRef getName() const { return Name; }
  llvm::StringRef getMangledName() const { return MangledName; }
  bool hasVL() const { return HasVL; }
  bool hasPolicy() const { return Scheme != SchemeNone; }
  bool hasPassthruOperand() const { return Scheme == HasPassthruOperand; }
  bool hasPolicyOperand() const { return Scheme == HasPolicyOperand; }
  bool hasUnMaskedOverloaded() const { return HasUnMaskedOverloaded; }
  bool hasBuiltinAlias() const { return HasBuiltinAlias; }
  bool hasManualCodegen() const { return !ManualCodegen.empty(); }
  bool isMasked() const { return IsMasked; }
  llvm::StringRef getIRName() const { return IRName; }
  llvm::StringRef getManualCodegen() const { return ManualCodegen; }
  PolicyScheme getPolicyScheme() const { return Scheme; }
  RISCVPredefinedMacroT getRISCVPredefinedMacros() const {
    return RISCVPredefinedMacros;
  }
  unsigned getNF() const { return NF; }
  const std::vector<int64_t> &getIntrinsicTypes() const {
    return IntrinsicTypes;
  }

  // Return the type string for a BUILTIN() macro in Builtins.def.
  std::string getBuiltinTypeStr() const;
};

} // end namespace RISCV

} // end namespace clang

#endif // CLANG_SUPPORT_RISCVVINTRINSICUTILS_H
