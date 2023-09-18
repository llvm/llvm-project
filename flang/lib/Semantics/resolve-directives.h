//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_RESOLVE_DIRECTIVES_H_
#define FORTRAN_SEMANTICS_RESOLVE_DIRECTIVES_H_

namespace Fortran::parser {
struct Name;
struct Program;
struct ProgramUnit;
} // namespace Fortran::parser

namespace Fortran::semantics {

class SemanticsContext;

// Name resolution for OpenACC and OpenMP directives
void ResolveAccParts(SemanticsContext &, const parser::ProgramUnit &);
void ResolveOmpParts(SemanticsContext &, const parser::ProgramUnit &);
void ResolveOmpTopLevelParts(SemanticsContext &, const parser::Program &);

} // namespace Fortran::semantics
#endif
