//== PseudoConstantAnalysis.h - Find Pseudo-constants in the AST -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks the usage of variables in a Decl body to see if they are
// never written to, implying that they constant. This is useful in static
// analysis to see if a developer might have intended a variable to be const.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PSEUDOCONSTANTANALYSIS
#define LLVM_CLANG_ANALYSIS_PSEUDOCONSTANTANALYSIS

#include "clang/AST/Stmt.h"

namespace clang {

class PseudoConstantAnalysis {
public:
  PseudoConstantAnalysis(const Stmt *DeclBody);
  ~PseudoConstantAnalysis();

  bool isPseudoConstant(const VarDecl *VD);
  bool wasReferenced(const VarDecl *VD);

private:
  void RunAnalysis();
  inline static const Decl *getDecl(const Expr *E);

  // for storing the result of analyzed ValueDecls
  void *NonConstantsImpl;
  void *UsedVarsImpl;

  const Stmt *DeclBody;
  bool Analyzed;
};

}

#endif
