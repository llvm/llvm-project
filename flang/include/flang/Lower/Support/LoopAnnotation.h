//===-- Lower/Support/LoopAnnotation.h -- loop annotation attrs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to lower Fortran `!dir$` loop directives to LLVM LoopAnnotationAttr.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_SUPPORT_LOOPANNOTATION_H
#define FORTRAN_LOWER_SUPPORT_LOOPANNOTATION_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <tuple>

namespace Fortran {
namespace parser {
struct CompilerDirective;
} // namespace parser

namespace lower {

/// Build an LLVM loop annotation attribute from Fortran compiler directives
/// associated with a loop. Returns null if \p dirs does not contain any
/// recognized loop directives.
mlir::LLVM::LoopAnnotationAttr genLoopAnnotationAttr(
    mlir::MLIRContext *context,
    llvm::ArrayRef<const Fortran::parser::CompilerDirective *> dirs);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_SUPPORT_LOOPANNOTATION_H
