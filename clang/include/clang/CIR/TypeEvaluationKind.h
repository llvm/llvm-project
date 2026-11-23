//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_TYPEEVALUATIONKIND_H
#define CLANG_CIR_TYPEEVALUATIONKIND_H

namespace cir {

// This is copied from clang/lib/CodeGen/CodeGenFunction.h.  That file (1) is
// not available as an include from ClangIR files, and (2) has lots of stuff
// that we don't want in ClangIR.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

} // namespace cir

#endif // CLANG_CIR_TYPEEVALUATIONKIND_H
