//== BodyFarm.h - Factory for conjuring up fake bodies -------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BodyFarm is a factory for creating faux implementations for functions/methods
// for analysis purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BODYFARM_H
#define LLVM_CLANG_ANALYSIS_BODYFARM_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class ASTContext;
class Decl;
class FunctionDecl;
class Stmt;
  
class BodyFarm {
public:
  BodyFarm(ASTContext &C) : C(C) {}
  
  /// Factory method for creating bodies for ordinary functions.
  Stmt *getBody(const FunctionDecl *D);
  
private:
  typedef llvm::DenseMap<const Decl *, llvm::Optional<Stmt *> > BodyMap;

  ASTContext &C;
  BodyMap Bodies;
};
}

#endif
