// BlkExprDeclBitVector.h - Dataflow types for Bitvector Analysis --*- C++ --*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides definition of dataflow types used by analyses such
// as LiveVariables and UninitializedValues.  The underlying dataflow values
// are implemented as bitvectors, but the definitions in this file include
// the necessary boilerplate to use with our dataflow framework.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STMTDECLBVDVAL_H
#define LLVM_CLANG_STMTDECLBVDVAL_H

#include "clang/Analysis/CFG.h"
#include "clang/AST/Decl.h" // for Decl* -> NamedDecl* conversion
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

  class Stmt;
  class ASTContext;

struct DeclBitVector_Types {

  class Idx {
    unsigned I;
  public:
    explicit Idx(unsigned i) : I(i) {}
    Idx() : I(~0U) {}

    bool isValid() const {
      return I != ~0U;
    }
    operator unsigned() const {
      assert (isValid());
      return I;
    }
  };

  //===--------------------------------------------------------------------===//
  // AnalysisDataTy - Whole-function meta data.
  //===--------------------------------------------------------------------===//

  class AnalysisDataTy {
  public:
    typedef llvm::DenseMap<const NamedDecl*, unsigned > DMapTy;
    typedef DMapTy::const_iterator decl_iterator;

  protected:
    DMapTy DMap;
    unsigned NDecls;

  public:

    AnalysisDataTy() : NDecls(0) {}
    virtual ~AnalysisDataTy() {}

    bool isTracked(const NamedDecl *SD) { return DMap.find(SD) != DMap.end(); }

    Idx getIdx(const NamedDecl *SD) const {
      DMapTy::const_iterator I = DMap.find(SD);
      return I == DMap.end() ? Idx() : Idx(I->second);
    }

    unsigned getNumDecls() const { return NDecls; }

    void Register(const NamedDecl *SD) {
      if (!isTracked(SD)) DMap[SD] = NDecls++;
    }

    decl_iterator begin_decl() const { return DMap.begin(); }
    decl_iterator end_decl() const { return DMap.end(); }
  };

  //===--------------------------------------------------------------------===//
  // ValTy - Dataflow value.
  //===--------------------------------------------------------------------===//

  class ValTy {
    llvm::BitVector DeclBV;
  public:

    void resetDeclValues(AnalysisDataTy& AD) {
      DeclBV.resize(AD.getNumDecls());
      DeclBV.reset();
    }

    void setDeclValues(AnalysisDataTy& AD) {
      DeclBV.resize(AD.getNumDecls());
      DeclBV.set();
    }

    void resetValues(AnalysisDataTy& AD) {
      resetDeclValues(AD);
    }

    bool operator==(const ValTy& RHS) const {
      assert (sizesEqual(RHS));
      return DeclBV == RHS.DeclBV;
    }

    void copyValues(const ValTy& RHS) { DeclBV = RHS.DeclBV; }

    llvm::BitVector::reference getBit(unsigned i) {
      return DeclBV[i];
    }

    bool getBit(unsigned i) const {
      return DeclBV[i];
    }

    llvm::BitVector::reference
    operator()(const NamedDecl *ND, const AnalysisDataTy& AD) {
      return getBit(AD.getIdx(ND));
    }

    bool operator()(const NamedDecl *ND, const AnalysisDataTy& AD) const {
      return getBit(AD.getIdx(ND));
    }

    llvm::BitVector::reference getDeclBit(unsigned i) { return DeclBV[i]; }
    const llvm::BitVector::reference getDeclBit(unsigned i) const {
      return const_cast<llvm::BitVector&>(DeclBV)[i];
    }

    ValTy& operator|=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      DeclBV |= RHS.DeclBV;
      return *this;
    }

    ValTy& operator&=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      DeclBV &= RHS.DeclBV;
      return *this;
    }

    ValTy& OrDeclBits(const ValTy& RHS) {
      return operator|=(RHS);
    }

    ValTy& AndDeclBits(const ValTy& RHS) {
      return operator&=(RHS);
    }

    bool sizesEqual(const ValTy& RHS) const {
      return DeclBV.size() == RHS.DeclBV.size();
    }
  };

  //===--------------------------------------------------------------------===//
  // Some useful merge operations.
  //===--------------------------------------------------------------------===//

  struct Union { void operator()(ValTy& Dst, ValTy& Src) { Dst |= Src; } };
  struct Intersect { void operator()(ValTy& Dst, ValTy& Src) { Dst &= Src; } };
};


struct StmtDeclBitVector_Types {

  //===--------------------------------------------------------------------===//
  // AnalysisDataTy - Whole-function meta data.
  //===--------------------------------------------------------------------===//

  class AnalysisDataTy : public DeclBitVector_Types::AnalysisDataTy {
    ASTContext *ctx;
    CFG* cfg;
  public:
    AnalysisDataTy() : ctx(0), cfg(0) {}
    virtual ~AnalysisDataTy() {}

    void setContext(ASTContext &c) { ctx = &c; }
    ASTContext &getContext() {
      assert(ctx && "ASTContext should not be NULL.");
      return *ctx;
    }

    void setCFG(CFG& c) { cfg = &c; }
    CFG& getCFG() { assert(cfg && "CFG should not be NULL."); return *cfg; }

    bool isTracked(const Stmt *S) { return cfg->isBlkExpr(S); }
    using DeclBitVector_Types::AnalysisDataTy::isTracked;

    unsigned getIdx(const Stmt *S) const {
      CFG::BlkExprNumTy I = cfg->getBlkExprNum(S);
      assert(I && "Stmtession not tracked for bitvector.");
      return I;
    }
    using DeclBitVector_Types::AnalysisDataTy::getIdx;

    unsigned getNumBlkExprs() const { return cfg->getNumBlkExprs(); }
  };

  //===--------------------------------------------------------------------===//
  // ValTy - Dataflow value.
  //===--------------------------------------------------------------------===//

  class ValTy : public DeclBitVector_Types::ValTy {
    llvm::BitVector BlkExprBV;
    typedef DeclBitVector_Types::ValTy ParentTy;

    static inline ParentTy& ParentRef(ValTy& X) {
      return static_cast<ParentTy&>(X);
    }

    static inline const ParentTy& ParentRef(const ValTy& X) {
      return static_cast<const ParentTy&>(X);
    }

  public:

    void resetBlkExprValues(AnalysisDataTy& AD) {
      BlkExprBV.resize(AD.getNumBlkExprs());
      BlkExprBV.reset();
    }

    void setBlkExprValues(AnalysisDataTy& AD) {
      BlkExprBV.resize(AD.getNumBlkExprs());
      BlkExprBV.set();
    }

    void resetValues(AnalysisDataTy& AD) {
      resetDeclValues(AD);
      resetBlkExprValues(AD);
    }

    void setValues(AnalysisDataTy& AD) {
      setDeclValues(AD);
      setBlkExprValues(AD);
    }

    bool operator==(const ValTy& RHS) const {
      return ParentRef(*this) == ParentRef(RHS)
          && BlkExprBV == RHS.BlkExprBV;
    }

    void copyValues(const ValTy& RHS) {
      ParentRef(*this).copyValues(ParentRef(RHS));
      BlkExprBV = RHS.BlkExprBV;
    }

    llvm::BitVector::reference
    operator()(const Stmt *S, const AnalysisDataTy& AD) {
      return BlkExprBV[AD.getIdx(S)];
    }
    const llvm::BitVector::reference
    operator()(const Stmt *S, const AnalysisDataTy& AD) const {
      return const_cast<ValTy&>(*this)(S,AD);
    }

    using DeclBitVector_Types::ValTy::operator();


    llvm::BitVector::reference getStmtBit(unsigned i) { return BlkExprBV[i]; }
    const llvm::BitVector::reference getStmtBit(unsigned i) const {
      return const_cast<llvm::BitVector&>(BlkExprBV)[i];
    }

    ValTy& OrBlkExprBits(const ValTy& RHS) {
      BlkExprBV |= RHS.BlkExprBV;
      return *this;
    }

    ValTy& AndBlkExprBits(const ValTy& RHS) {
      BlkExprBV &= RHS.BlkExprBV;
      return *this;
    }

    ValTy& operator|=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      ParentRef(*this) |= ParentRef(RHS);
      BlkExprBV |= RHS.BlkExprBV;
      return *this;
    }

    ValTy& operator&=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      ParentRef(*this) &= ParentRef(RHS);
      BlkExprBV &= RHS.BlkExprBV;
      return *this;
    }

    bool sizesEqual(const ValTy& RHS) const {
      return ParentRef(*this).sizesEqual(ParentRef(RHS))
          && BlkExprBV.size() == RHS.BlkExprBV.size();
    }
  };

  //===--------------------------------------------------------------------===//
  // Some useful merge operations.
  //===--------------------------------------------------------------------===//

  struct Union { void operator()(ValTy& Dst, ValTy& Src) { Dst |= Src; } };
  struct Intersect { void operator()(ValTy& Dst, ValTy& Src) { Dst &= Src; } };

};
} // end namespace clang

#endif
