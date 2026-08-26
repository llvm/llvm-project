//===-- lib/Semantics/resolve-labels.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_RESOLVE_LABELS_H_
#define FORTRAN_SEMANTICS_RESOLVE_LABELS_H_

namespace Fortran::parser {
struct Program;
}

namespace Fortran::semantics {
class SemanticsContext;

/// \brief Analyze the labels in the program: check them, and record the
/// statements that a branch may name so that lowering can consult the same
/// classification rather than deriving it again.
/// \param context   semantic context for errors
/// \param program    the parse tree of the program
/// \return true, iff the program's labels pass semantics checks
bool AnalyzeLabels(SemanticsContext &context, const parser::Program &program);
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_RESOLVE_LABELS_H_
