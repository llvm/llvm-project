//===- Target.h - target specific details -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTMIZER_CODEGEN_TARGET_H
#define FORTRAN_OPTMIZER_CODEGEN_TARGET_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <tuple>
#include <vector>

namespace mlir {
class DataLayout;
}

namespace fir {

namespace details {
/// Extra information about how to marshal an argument or return value that
/// modifies a signature per a particular ABI's calling convention.
/// Note: llvm::Attribute is not used directly, because its use depends on an
/// LLVMContext.
class Attributes {
public:
  enum class IntegerExtension { None, Zero, Sign };

  Attributes(unsigned short alignment = 0, bool byval = false,
             bool sret = false, bool append = false,
             IntegerExtension intExt = IntegerExtension::None)
      : alignment{alignment}, byval{byval}, sret{sret}, append{append},
        intExt{intExt} {}

  unsigned getAlignment() const { return alignment; }
  bool hasAlignment() const { return alignment != 0; }
  bool isByVal() const { return byval; }
  bool isSRet() const { return sret; }
  bool isAppend() const { return append; }
  bool isZeroExt() const { return intExt == IntegerExtension::Zero; }
  bool isSignExt() const { return intExt == IntegerExtension::Sign; }
  llvm::StringRef getIntExtensionAttrName() const;

private:
  unsigned short alignment{};
  bool byval : 1;
  bool sret : 1;
  bool append : 1;
  IntegerExtension intExt;
};

} // namespace details

/// Some details of how to represent certain features depend on the target and
/// ABI that is being used.  These specifics are captured here and guide the
/// lowering of FIR to LLVM-IR dialect.
class CodeGenSpecifics {
public:
  using Attributes = details::Attributes;
  using TypeAndAttr = std::tuple<mlir::Type, Attributes>;
  using Marshalling = std::vector<TypeAndAttr>;

  static std::unique_ptr<CodeGenSpecifics>
  get(mlir::MLIRContext *ctx, llvm::Triple &&trp, KindMapping &&kindMap,
      llvm::StringRef targetCPU, mlir::LLVM::TargetFeaturesAttr targetFeatures,
      const mlir::DataLayout &dl);

  static TypeAndAttr getTypeAndAttr(mlir::Type t) { return TypeAndAttr{t, {}}; }

  CodeGenSpecifics(mlir::MLIRContext *ctx, llvm::Triple &&trp,
                   KindMapping &&kindMap, llvm::StringRef targetCPU,
                   mlir::LLVM::TargetFeaturesAttr targetFeatures,
                   const mlir::DataLayout &dl)
      : context{*ctx}, triple{std::move(trp)}, kindMap{std::move(kindMap)},
        targetCPU{targetCPU}, targetFeatures{targetFeatures}, dataLayout{&dl} {}
  CodeGenSpecifics() = delete;
  virtual ~CodeGenSpecifics() {}

  /// Type presentation of a `complex<ele>` type value in memory.
  virtual mlir::Type complexMemoryType(mlir::Type eleTy) const = 0;

  /// Type representation of a `complex<eleTy>` type argument when passed by
  /// value. An argument value may need to be passed as a (safe) reference
  /// argument.
  virtual Marshalling complexArgumentType(mlir::Location loc,
                                          mlir::Type eleTy) const = 0;

  /// Type representation of a `complex<eleTy>` type return value. Such a return
  /// value may need to be converted to a hidden reference argument.
  virtual Marshalling complexReturnType(mlir::Location loc,
                                        mlir::Type eleTy) const = 0;

  /// Type presentation of a `boxchar<n>` type value in memory.
  virtual mlir::Type boxcharMemoryType(mlir::Type eleTy) const = 0;

  /// Type representation of a `fir.type<T>` type argument when passed by
  /// value. It may have to be split into several arguments, or be passed
  /// as a byval reference argument (on the stack).
  virtual Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType recTy,
                     const Marshalling &previousArguments) const = 0;

  /// Type representation of a `boxchar<n>` type argument when passed by value.
  /// An argument value may need to be passed as a (safe) reference argument.
  ///
  /// A function that returns a `boxchar<n>` type value must already have
  /// converted that return value to a parameter decorated with the 'sret'
  /// Attribute (https://llvm.org/docs/LangRef.html#parameter-attributes).
  /// This requirement is in keeping with Fortran semantics, which require the
  /// caller to allocate the space for the return CHARACTER value and pass
  /// a pointer and the length of that space (a boxchar) to the called function.
  virtual Marshalling boxcharArgumentType(mlir::Type eleTy,
                                          bool sret = false) const = 0;

  // Compute ABI rules for an integer argument of the given mlir::IntegerType
  // \p argTy. Note that this methods is supposed to be called for
  // arguments passed by value not via reference, e.g. the 'i1' argument here:
  //   declare i1 @_FortranAioOutputLogical(ptr, i1)
  //
  // \p loc is the location of the operation using/specifying the argument.
  //
  // Currently, the only supported marshalling is whether the argument
  // should be zero or sign extended.
  //
  // The zero/sign extension is especially important to comply with the ABI
  // used by C/C++ compiler that builds Fortran runtime. As in the above
  // example the callee will expect the caller to zero extend the second
  // argument up to the size of the C/C++'s 'int' type.
  // The corresponding handling in clang is done in
  // DefaultABIInfo::classifyArgumentType(), and the logic may brielfy
  // be explained as some sort of extension is required if the integer
  // type is shorter than the size of 'int' for the target.
  // The related code is located in ASTContext::isPromotableIntegerType()
  // and ABIInfo::isPromotableIntegerTypeForABI().
  // In particular, the latter returns 'true' for 'bool', several kinds
  // of 'char', 'short', 'wchar' and enumerated types.
  // The type of the extensions (zero or sign) depends on the signedness
  // of the original language type.
  //
  // It is not clear how to handle signless integer types.
  // From the point of Fortran-C interface all supported integer types
  // seem to be signed except for CFI_type_Bool/bool that is supported
  // via signless 'i1', but that is treated as unsigned type by clang
  // (e.g. 'bool' arguments are using 'zeroext' ABI).
  virtual Marshalling integerArgumentType(mlir::Location loc,
                                          mlir::IntegerType argTy) const = 0;

  // By default, integer argument and return values use the same
  // zero/sign extension rules.
  virtual Marshalling integerReturnType(mlir::Location loc,
                                        mlir::IntegerType argTy) const = 0;

  // Returns width in bits of C/C++ 'int' type size.
  virtual unsigned char getCIntTypeWidth() const = 0;

  llvm::StringRef getTargetCPU() const { return targetCPU; }

  mlir::LLVM::TargetFeaturesAttr getTargetFeatures() const {
    return targetFeatures;
  }

  const mlir::DataLayout &getDataLayout() const {
    assert(dataLayout && "dataLayout must be set");
    return *dataLayout;
  }

protected:
  mlir::MLIRContext &context;
  llvm::Triple triple;
  KindMapping kindMap;
  llvm::StringRef targetCPU;
  mlir::LLVM::TargetFeaturesAttr targetFeatures;
  const mlir::DataLayout *dataLayout = nullptr;
};

} // namespace fir

#endif // FORTRAN_OPTMIZER_CODEGEN_TARGET_H
