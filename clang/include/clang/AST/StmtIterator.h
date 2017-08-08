//===--- StmtIterator.h - Iterators for Statements --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StmtIterator and ConstStmtIterator classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTITERATOR_H
#define LLVM_CLANG_AST_STMTITERATOR_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

namespace clang {

class Stmt;
class Decl;
class VariableArrayType;

class StmtIteratorBase {
protected:
  enum { StmtMode = 0x0, SizeOfTypeVAMode = 0x1, DeclGroupMode = 0x2,
         Flags = 0x3 };
  
  union {
    Stmt **stmt;
    Decl **DGI;
  };
  uintptr_t RawVAPtr;
  Decl **DGE;
  
  bool inDeclGroup() const {
    return (RawVAPtr & Flags) == DeclGroupMode;
  }

  bool inSizeOfTypeVA() const {
    return (RawVAPtr & Flags) == SizeOfTypeVAMode;
  }

  bool inStmt() const {
    return (RawVAPtr & Flags) == StmtMode;
  }

  const VariableArrayType *getVAPtr() const {
    return reinterpret_cast<const VariableArrayType*>(RawVAPtr & ~Flags);
  }

  void setVAPtr(const VariableArrayType *P) {
    assert (inDeclGroup() || inSizeOfTypeVA());
    RawVAPtr = reinterpret_cast<uintptr_t>(P) | (RawVAPtr & Flags);
  }

  void NextDecl(bool ImmediateAdvance = true);
  bool HandleDecl(Decl* D);
  void NextVA();

  Stmt*& GetDeclExpr() const;

  StmtIteratorBase(Stmt **s) : stmt(s), RawVAPtr(0) {}
  StmtIteratorBase(const VariableArrayType *t);
  StmtIteratorBase(Decl **dgi, Decl **dge);
  StmtIteratorBase() : stmt(nullptr), RawVAPtr(0) {}
};


template <typename DERIVED, typename REFERENCE>
class StmtIteratorImpl : public StmtIteratorBase,
                         public std::iterator<std::forward_iterator_tag,
                                              REFERENCE, ptrdiff_t,
                                              REFERENCE, REFERENCE> {
protected:
  StmtIteratorImpl(const StmtIteratorBase& RHS) : StmtIteratorBase(RHS) {}
public:
  StmtIteratorImpl() {}
  StmtIteratorImpl(Stmt **s) : StmtIteratorBase(s) {}
  StmtIteratorImpl(Decl **dgi, Decl **dge) : StmtIteratorBase(dgi, dge) {}
  StmtIteratorImpl(const VariableArrayType *t) : StmtIteratorBase(t) {}

  DERIVED& operator++() {
    if (inStmt())
      ++stmt;
    else if (getVAPtr())
      NextVA();
    else
      NextDecl();

    return static_cast<DERIVED&>(*this);
  }

  DERIVED operator++(int) {
    DERIVED tmp = static_cast<DERIVED&>(*this);
    operator++();
    return tmp;
  }

  bool operator==(const DERIVED& RHS) const {
    return stmt == RHS.stmt && DGI == RHS.DGI && RawVAPtr == RHS.RawVAPtr;
  }

  bool operator!=(const DERIVED& RHS) const {
    return stmt != RHS.stmt || DGI != RHS.DGI || RawVAPtr != RHS.RawVAPtr;
  }

  REFERENCE operator*() const {
    return inStmt() ? *stmt : GetDeclExpr();
  }

  REFERENCE operator->() const { return operator*(); }
};

struct ConstStmtIterator;

struct StmtIterator : public StmtIteratorImpl<StmtIterator,Stmt*&> {
  explicit StmtIterator() : StmtIteratorImpl<StmtIterator,Stmt*&>() {}

  StmtIterator(Stmt** S) : StmtIteratorImpl<StmtIterator,Stmt*&>(S) {}

  StmtIterator(Decl** dgi, Decl** dge)
   : StmtIteratorImpl<StmtIterator,Stmt*&>(dgi, dge) {}

  StmtIterator(const VariableArrayType *t)
    : StmtIteratorImpl<StmtIterator,Stmt*&>(t) {}

private:
  StmtIterator(const StmtIteratorBase &RHS)
      : StmtIteratorImpl<StmtIterator, Stmt *&>(RHS) {}

  inline friend StmtIterator
  cast_away_const(const ConstStmtIterator &RHS);
};

struct ConstStmtIterator : public StmtIteratorImpl<ConstStmtIterator,
                                                   const Stmt*> {
  explicit ConstStmtIterator() :
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>() {}

  ConstStmtIterator(const StmtIterator& RHS) :
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>(RHS) {}

  ConstStmtIterator(Stmt * const *S)
      : StmtIteratorImpl<ConstStmtIterator, const Stmt *>(
            const_cast<Stmt **>(S)) {}
};

inline StmtIterator cast_away_const(const ConstStmtIterator &RHS) {
  return RHS;
}
} // end namespace clang

#endif
