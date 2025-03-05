//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H

#include "CIRGenTypeCache.h"

#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"

namespace clang::CIRGen {

class CIRGenBuilderTy : public cir::CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;

public:
  CIRGenBuilderTy(mlir::MLIRContext &mlirContext, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(mlirContext), typeCache(tc) {}

  cir::LongDoubleType getLongDoubleTy(const llvm::fltSemantics &format) const {
    if (&format == &llvm::APFloat::IEEEdouble())
      return cir::LongDoubleType::get(getContext(), typeCache.DoubleTy);
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return cir::LongDoubleType::get(getContext(), typeCache.FP80Ty);
    if (&format == &llvm::APFloat::IEEEquad())
      return cir::LongDoubleType::get(getContext(), typeCache.FP128Ty);
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("NYI: PPC double-double format for long double");
    llvm_unreachable("Unsupported format for long double");
  }
};

} // namespace clang::CIRGen

#endif
