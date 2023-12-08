//===-- lib/Semantics/rewrite-parse-tree.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_REWRITE_PARSE_TREE_H_
#define FORTRAN_SEMANTICS_REWRITE_PARSE_TREE_H_

namespace Fortran::parser {
class Messages;
struct Program;
} // namespace Fortran::parser
namespace Fortran::semantics {
class SemanticsContext;
}

namespace Fortran::semantics {
bool RewriteParseTree(SemanticsContext &, parser::Program &);
}

#endif // FORTRAN_SEMANTICS_REWRITE_PARSE_TREE_H_
