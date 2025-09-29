//===------- Optimizer/Transforms/MIFOpToLLVMConversion.h -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_

#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;
}

namespace mif {

// Default prefix for subroutines of PRIF compiled with LLVM
#define PRIFNAME_SUB(fmt)                                                      \
  []() {                                                                       \
    std::ostringstream oss;                                                    \
    oss << "prif_" << fmt;                                                     \
    return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());             \
  }()

#define PRIF_STAT_TYPE builder.getRefType(builder.getI32Type())
#define PRIF_ERRMSG_TYPE                                                       \
  fir::BoxType::get(fir::CharacterType::get(builder.getContext(), 1,           \
                                            fir::CharacterType::unknownLen()))

/// Patterns that convert MIF operations to runtime calls.
void populateMIFOpConversionPatterns(fir::LLVMTypeConverter &converter,
                                     mlir::RewritePatternSet &patterns);
} // namespace mif

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
