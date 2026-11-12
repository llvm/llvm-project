//===-- lib/Semantics/canonicalize-do.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CANONICALIZE_DO_H_
#define FORTRAN_SEMANTICS_CANONICALIZE_DO_H_

// Converts a LabelDo followed by a sequence of ExecutableConstructs (perhaps
// logically nested) into the more structured DoConstruct (explicitly nested)
namespace Fortran::parser {
struct Program;
bool CanonicalizeDo(Program &program);
} // namespace Fortran::parser

#endif // FORTRAN_SEMANTICS_CANONICALIZE_DO_H_
