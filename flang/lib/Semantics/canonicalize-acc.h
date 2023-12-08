//===-- lib/Semantics/canonicalize-acc.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CANONICALIZE_ACC_H_
#define FORTRAN_SEMANTICS_CANONICALIZE_ACC_H_

namespace Fortran::parser {
struct Program;
class Messages;
} // namespace Fortran::parser

namespace Fortran::semantics {
bool CanonicalizeAcc(parser::Messages &messages, parser::Program &program);
}

#endif // FORTRAN_SEMANTICS_CANONICALIZE_ACC_H_
