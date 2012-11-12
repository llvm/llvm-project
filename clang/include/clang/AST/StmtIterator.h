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

#ifndef LLVM_CLANG_AST_STMT_ITR_H
#define LLVM_CLANG_AST_STMT_ITR_H

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
  enum { DeclMode = 0x1, SizeOfTypeVAMode = 0x2, DeclGroupMode = 0x3,
         Flags = 0x3 };
  
  Stmt **stmt;
  union { Decl *decl; Decl **DGI; };
  uintptr_t RawVAPtr;
  Decl **DGE;
  
  bool inDecl() const {
    return (RawVAPtr & Flags) == DeclMode;
  }

  bool inDeclGroup() const {
    return (RawVAPtr & Flags) == DeclGroupMode;
  }

  bool inSizeOfTypeVA() const {
    return (RawVAPtr & Flags) == SizeOfTypeVAMode;
  }

  bool inStmt() const {
    return (RawVAPtr & Flags) == 0;
  }

  const VariableArrayType *getVAPtr() const {
    return reinterpret_cast<const VariableArrayType*>(RawVAPtr & ~Flags);
  }

  void setVAPtr(const VariableArrayType *P) {
    assert (inDecl() || inDeclGroup() || inSizeOfTypeVA());
    RawVAPtr = reinterpret_cast<uintptr_t>(P) | (RawVAPtr & Flags);
  }

  void NextDecl(bool ImmediateAdvance = true);
  bool HandleDecl(Decl* D);
  void NextVA();

  Stmt*& GetDeclExpr() const;

  StmtIteratorBase(Stmt **s) : stmt(s), decl(0), RawVAPtr(0) {}
  StmtIteratorBase(Decl *d, Stmt **s);
  StmtIteratorBase(const VariableArrayType *t);
  StmtIteratorBase(Decl **dgi, Decl **dge);
  StmtIteratorBase() : stmt(0), decl(0), RawVAPtr(0) {}
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
  StmtIteratorImpl(Decl *d, Stmt **s) : StmtIteratorBase(d, s) {}
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
    return stmt == RHS.stmt && decl == RHS.decl && RawVAPtr == RHS.RawVAPtr;
  }

  bool operator!=(const DERIVED& RHS) const {
    return stmt != RHS.stmt || decl != RHS.decl || RawVAPtr != RHS.RawVAPtr;
  }

  REFERENCE operator*() const {
    return (REFERENCE) (inStmt() ? *stmt : GetDeclExpr());
  }

  REFERENCE operator->() const { return operator*(); }
};

struct StmtIterator : public StmtIteratorImpl<StmtIterator,Stmt*&> {
  explicit StmtIterator() : StmtIteratorImpl<StmtIterator,Stmt*&>() {}

  StmtIterator(Stmt** S) : StmtIteratorImpl<StmtIterator,Stmt*&>(S) {}

  StmtIterator(Decl** dgi, Decl** dge)
   : StmtIteratorImpl<StmtIterator,Stmt*&>(dgi, dge) {}

  StmtIterator(const VariableArrayType *t)
    : StmtIteratorImpl<StmtIterator,Stmt*&>(t) {}

  StmtIterator(Decl* D, Stmt **s = 0)
    : StmtIteratorImpl<StmtIterator,Stmt*&>(D, s) {}
};

struct ConstStmtIterator : public StmtIteratorImpl<ConstStmtIterator,
                                                   const Stmt*> {
  explicit ConstStmtIterator() :
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>() {}

  ConstStmtIterator(const StmtIterator& RHS) :
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>(RHS) {}
};

/// A range of statement iterators.
///
/// This class provides some extra functionality beyond std::pair
/// in order to allow the following idiom:
///   for (StmtRange range = stmt->children(); range; ++range)
struct StmtRange : std::pair<StmtIterator,StmtIterator> {
  StmtRange() {}
  StmtRange(const StmtIterator &begin, const StmtIterator &end)
    : std::pair<StmtIterator,StmtIterator>(begin, end) {}

  bool empty() const { return first == second; }
  operator bool() const { return !empty(); }

  Stmt *operator->() const { return first.operator->(); }
  Stmt *&operator*() const { return first.operator*(); }

  StmtRange &operator++() {
    assert(!empty() && "incrementing on empty range");
    ++first;
    return *this;
  }

  StmtRange operator++(int) {
    assert(!empty() && "incrementing on empty range");
    StmtRange copy = *this;
    ++first;
    return copy;
  }

  friend const StmtIterator &begin(const StmtRange &range) {
    return range.first;
  }
  friend const StmtIterator &end(const StmtRange &range) {
    return range.second;
  }
};

/// A range of const statement iterators.
///
/// This class provides some extra functionality beyond std::pair
/// in order to allow the following idiom:
///   for (ConstStmtRange range = stmt->children(); range; ++range)
struct ConstStmtRange : std::pair<ConstStmtIterator,ConstStmtIterator> {
  ConstStmtRange() {}
  ConstStmtRange(const ConstStmtIterator &begin,
                 const ConstStmtIterator &end)
    : std::pair<ConstStmtIterator,ConstStmtIterator>(begin, end) {}
  ConstStmtRange(const StmtRange &range)
    : std::pair<ConstStmtIterator,ConstStmtIterator>(range.first, range.second)
  {}
  ConstStmtRange(const StmtIterator &begin, const StmtIterator &end)
    : std::pair<ConstStmtIterator,ConstStmtIterator>(begin, end) {}

  bool empty() const { return first == second; }
  operator bool() const { return !empty(); }

  const Stmt *operator->() const { return first.operator->(); }
  const Stmt *operator*() const { return first.operator*(); }

  ConstStmtRange &operator++() {
    assert(!empty() && "incrementing on empty range");
    ++first;
    return *this;
  }

  ConstStmtRange operator++(int) {
    assert(!empty() && "incrementing on empty range");
    ConstStmtRange copy = *this;
    ++first;
    return copy;
  }

  friend const ConstStmtIterator &begin(const ConstStmtRange &range) {
    return range.first;
  }
  friend const ConstStmtIterator &end(const ConstStmtRange &range) {
    return range.second;
  }
};

} // end namespace clang

#endif
