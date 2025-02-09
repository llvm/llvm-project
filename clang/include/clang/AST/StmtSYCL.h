//===- StmtSYCL.h - Classes for SYCL kernel calls ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines SYCL AST classes used to represent calls to SYCL kernels.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTSYCL_H
#define LLVM_CLANG_AST_STMTSYCL_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for SYCL kernel calls.
//===----------------------------------------------------------------------===//

/// SYCLKernelCallStmt represents the transformation that is applied to the body
/// of a function declared with the sycl_kernel_entry_point attribute. The body
/// of such a function specifies the statements to be executed on a SYCL device
/// to invoke a SYCL kernel with a particular set of kernel arguments. The
/// SYCLKernelCallStmt associates an original statement (the compound statement
/// that is the function body) with an OutlinedFunctionDecl that holds the
/// kernel parameters and the transformed body. During code generation, the
/// OutlinedFunctionDecl is used to emit an offload kernel entry point suitable
/// for invocation from a SYCL library implementation. If executed, the
/// SYCLKernelCallStmt behaves as a no-op; no code generation is performed for
/// it.
class SYCLKernelCallStmt : public Stmt {
  friend class ASTStmtReader;
  friend class ASTStmtWriter;

private:
  Stmt *OriginalStmt = nullptr;
  OutlinedFunctionDecl *OFDecl = nullptr;

public:
  /// Construct a SYCL kernel call statement.
  SYCLKernelCallStmt(CompoundStmt *CS, OutlinedFunctionDecl *OFD)
      : Stmt(SYCLKernelCallStmtClass), OriginalStmt(CS), OFDecl(OFD) {}

  /// Construct an empty SYCL kernel call statement.
  SYCLKernelCallStmt(EmptyShell Empty) : Stmt(SYCLKernelCallStmtClass, Empty) {}

  /// Retrieve the model statement.
  CompoundStmt *getOriginalStmt() { return cast<CompoundStmt>(OriginalStmt); }
  const CompoundStmt *getOriginalStmt() const {
    return cast<CompoundStmt>(OriginalStmt);
  }
  void setOriginalStmt(CompoundStmt *CS) { OriginalStmt = CS; }

  /// Retrieve the outlined function declaration.
  OutlinedFunctionDecl *getOutlinedFunctionDecl() { return OFDecl; }
  const OutlinedFunctionDecl *getOutlinedFunctionDecl() const { return OFDecl; }

  /// Set the outlined function declaration.
  void setOutlinedFunctionDecl(OutlinedFunctionDecl *OFD) { OFDecl = OFD; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getOriginalStmt()->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {
    return getOriginalStmt()->getEndLoc();
  }

  SourceRange getSourceRange() const LLVM_READONLY {
    return getOriginalStmt()->getSourceRange();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SYCLKernelCallStmtClass;
  }

  child_range children() {
    return child_range(&OriginalStmt, &OriginalStmt + 1);
  }

  const_child_range children() const {
    return const_child_range(&OriginalStmt, &OriginalStmt + 1);
  }
};

} // end namespace clang

#endif // LLVM_CLANG_AST_STMTSYCL_H
