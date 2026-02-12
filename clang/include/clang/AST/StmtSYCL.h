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
/// that is the function body) with a kernel launch statement to execute on a
/// SYCL host and an OutlinedFunctionDecl that holds the kernel parameters and
/// the transformed body to execute on a SYCL device. During code generation,
/// the OutlinedFunctionDecl is used to emit an offload kernel entry point
/// suitable for invocation from a SYCL library implementation.
class SYCLKernelCallStmt : public Stmt {
  friend class ASTStmtReader;
  friend class ASTStmtWriter;

private:
  Stmt *OriginalStmt = nullptr;
  Stmt *KernelLaunchStmt = nullptr;
  OutlinedFunctionDecl *OFDecl = nullptr;

public:
  /// Construct a SYCL kernel call statement.
  SYCLKernelCallStmt(CompoundStmt *CS, Stmt *S, OutlinedFunctionDecl *OFD)
      : Stmt(SYCLKernelCallStmtClass), OriginalStmt(CS), KernelLaunchStmt(S),
        OFDecl(OFD) {}

  /// Construct an empty SYCL kernel call statement.
  SYCLKernelCallStmt(EmptyShell Empty) : Stmt(SYCLKernelCallStmtClass, Empty) {}

  CompoundStmt *getOriginalStmt() { return cast<CompoundStmt>(OriginalStmt); }
  const CompoundStmt *getOriginalStmt() const {
    return cast<CompoundStmt>(OriginalStmt);
  }

  void setOriginalStmt(CompoundStmt *CS) { OriginalStmt = CS; }

  Stmt *getKernelLaunchStmt() { return KernelLaunchStmt; }
  const Stmt *getKernelLaunchStmt() const { return KernelLaunchStmt; }

  void setKernelLaunchStmt(Stmt *S) { KernelLaunchStmt = S; }

  OutlinedFunctionDecl *getOutlinedFunctionDecl() { return OFDecl; }
  const OutlinedFunctionDecl *getOutlinedFunctionDecl() const { return OFDecl; }

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

// UnresolvedSYCLKernelCallStmt represents an invocation of a SYCL kernel in
// a dependent context for which lookup of the sycl_kernel_launch identifier
// cannot be performed. These statements are transformed to SYCLKernelCallStmt
// during template instantiation.
class UnresolvedSYCLKernelCallStmt : public Stmt {
  friend class ASTStmtReader;
  friend class ASTStmtWriter;

private:
  Stmt *OriginalStmt = nullptr;
  // KernelLaunchIdExpr stores an UnresolvedLookupExpr or UnresolvedMemberExpr
  // corresponding to the SYCL kernel launch function for which a call
  // will be synthesized during template instantiation.
  Expr *KernelLaunchIdExpr = nullptr;

  UnresolvedSYCLKernelCallStmt(CompoundStmt *CS, Expr *IdExpr)
      : Stmt(UnresolvedSYCLKernelCallStmtClass), OriginalStmt(CS),
        KernelLaunchIdExpr(IdExpr) {}

  /// Set the original statement.
  void setOriginalStmt(CompoundStmt *CS) { OriginalStmt = CS; }

  /// Set the kernel launch ID expression.
  void setKernelLaunchIdExpr(Expr *IdExpr) { KernelLaunchIdExpr = IdExpr; }

public:
  static UnresolvedSYCLKernelCallStmt *Create(const ASTContext &C,
                                              CompoundStmt *CS, Expr *IdExpr) {
    return new (C) UnresolvedSYCLKernelCallStmt(CS, IdExpr);
  }

  static UnresolvedSYCLKernelCallStmt *CreateEmpty(const ASTContext &C) {
    return new (C) UnresolvedSYCLKernelCallStmt(nullptr, nullptr);
  }

  /// Retrieve the original statement.
  CompoundStmt *getOriginalStmt() { return cast<CompoundStmt>(OriginalStmt); }
  const CompoundStmt *getOriginalStmt() const {
    return cast<CompoundStmt>(OriginalStmt);
  }

  /// Retrieve the kernel launch ID expression.
  Expr *getKernelLaunchIdExpr() { return KernelLaunchIdExpr; }
  const Expr *getKernelLaunchIdExpr() const { return KernelLaunchIdExpr; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getOriginalStmt()->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {
    return getOriginalStmt()->getEndLoc();
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == UnresolvedSYCLKernelCallStmtClass;
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
