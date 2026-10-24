//===----- SemaBPF.h ------- BPF target-specific routines -----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to BPF.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMABPF_H
#define LLVM_CLANG_SEMA_SEMABPF_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/MapVector.h"

namespace clang {
class ParsedAttr;

class SemaBPF : public SemaBase {
public:
  SemaBPF(Sema &S);

  bool CheckBPFBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);

  void handlePreserveAIRecord(RecordDecl *RD);
  void handlePreserveAccessIndexAttr(Decl *D, const ParsedAttr &AL);
  void DiagnoseAnonymousPreserveAIEnums();
  Expr *BuildPreserveAIEnumeratorRef(DeclRefExpr *E);
  static bool isPreserveAIEnumeratorRef(const Expr *E);

private:
  llvm::MapVector<EnumDecl *, const BPFPreserveAccessIndexAttr *>
      AnonPreserveAIEnums;
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMABPF_H
