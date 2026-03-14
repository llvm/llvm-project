//===-- lib/Semantics/check-declarations.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Static declaration checks

#ifndef FORTRAN_SEMANTICS_CHECK_DECLARATIONS_H_
#define FORTRAN_SEMANTICS_CHECK_DECLARATIONS_H_
namespace Fortran::semantics {
class SemanticsContext;
void CheckDeclarations(SemanticsContext &);
} // namespace Fortran::semantics
#endif
