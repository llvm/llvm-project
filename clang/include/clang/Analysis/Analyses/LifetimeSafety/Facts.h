//===- Facts.h - Lifetime Analysis Facts and Fact Manager ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Facts, which are atomic lifetime-relevant events (such as
// loan issuance, loan expiration, origin flow, and use), and the FactManager,
// which manages the storage and retrieval of facts for each CFG block.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H

#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

namespace clang::lifetimes::internal {
/// An abstract base class for a single, atomic lifetime-relevant event.
class Fact {

public:
  enum class Kind : uint8_t {
    /// A new loan is issued from a borrow expression (e.g., &x).
    Issue,
    /// A loan expires as its underlying storage is freed (e.g., variable goes
    /// out of scope).
    Expire,
    /// An origin is propagated from a source to a destination (e.g., p = q).
    /// This can also optionally kill the destination origin before flowing into
    /// it. Otherwise, the source's loan set is merged into the destination's
    /// loan set.
    OriginFlow,
    /// An origin escapes the function by flowing into the return value.
    ReturnOfOrigin,
    /// An origin is used (eg. appears as l-value expression like DeclRefExpr).
    Use,
    /// A marker for a specific point in the code, for testing.
    TestPoint,
  };

private:
  Kind K;

protected:
  Fact(Kind K) : K(K) {}

public:
  virtual ~Fact() = default;
  Kind getKind() const { return K; }

  template <typename T> const T *getAs() const {
    if (T::classof(this))
      return static_cast<const T *>(this);
    return nullptr;
  }

  virtual void dump(llvm::raw_ostream &OS, const LoanManager &,
                    const OriginManager &) const;
};

/// A `ProgramPoint` identifies a location in the CFG by pointing to a specific
/// `Fact`. identified by a lifetime-related event (`Fact`).
///
/// A `ProgramPoint` has "after" semantics: it represents the location
/// immediately after its corresponding `Fact`.
using ProgramPoint = const Fact *;

class IssueFact : public Fact {
  LoanID LID;
  OriginID OID;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Issue; }

  IssueFact(LoanID LID, OriginID OID) : Fact(Kind::Issue), LID(LID), OID(OID) {}
  LoanID getLoanID() const { return LID; }
  OriginID getOriginID() const { return OID; }
  void dump(llvm::raw_ostream &OS, const LoanManager &LM,
            const OriginManager &OM) const override;
};

class ExpireFact : public Fact {
  LoanID LID;
  SourceLocation ExpiryLoc;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Expire; }

  ExpireFact(LoanID LID, SourceLocation ExpiryLoc)
      : Fact(Kind::Expire), LID(LID), ExpiryLoc(ExpiryLoc) {}

  LoanID getLoanID() const { return LID; }
  SourceLocation getExpiryLoc() const { return ExpiryLoc; }

  void dump(llvm::raw_ostream &OS, const LoanManager &LM,
            const OriginManager &) const override;
};

class OriginFlowFact : public Fact {
  OriginID OIDDest;
  OriginID OIDSrc;
  // True if the destination origin should be killed (i.e., its current loans
  // cleared) before the source origin's loans are flowed into it.
  bool KillDest;

public:
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::OriginFlow;
  }

  OriginFlowFact(OriginID OIDDest, OriginID OIDSrc, bool KillDest)
      : Fact(Kind::OriginFlow), OIDDest(OIDDest), OIDSrc(OIDSrc),
        KillDest(KillDest) {}

  OriginID getDestOriginID() const { return OIDDest; }
  OriginID getSrcOriginID() const { return OIDSrc; }
  bool getKillDest() const { return KillDest; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override;
};

class ReturnOfOriginFact : public Fact {
  OriginID OID;

public:
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::ReturnOfOrigin;
  }

  ReturnOfOriginFact(OriginID OID) : Fact(Kind::ReturnOfOrigin), OID(OID) {}
  OriginID getReturnedOriginID() const { return OID; }
  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override;
};

class UseFact : public Fact {
  const Expr *UseExpr;
  // True if this use is a write operation (e.g., left-hand side of assignment).
  // Write operations are exempted from use-after-free checks.
  bool IsWritten = false;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Use; }

  UseFact(const Expr *UseExpr) : Fact(Kind::Use), UseExpr(UseExpr) {}

  OriginID getUsedOrigin(const OriginManager &OM) const {
    // TODO: Remove const cast and make OriginManager::get as const.
    return const_cast<OriginManager &>(OM).get(*UseExpr);
  }
  const Expr *getUseExpr() const { return UseExpr; }
  void markAsWritten() { IsWritten = true; }
  bool isWritten() const { return IsWritten; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override;
};

/// A dummy-fact used to mark a specific point in the code for testing.
/// It is generated by recognizing a `void("__lifetime_test_point_...")` cast.
class TestPointFact : public Fact {
  StringRef Annotation;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::TestPoint; }

  explicit TestPointFact(StringRef Annotation)
      : Fact(Kind::TestPoint), Annotation(Annotation) {}

  StringRef getAnnotation() const { return Annotation; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &) const override;
};

class FactManager {
public:
  llvm::ArrayRef<const Fact *> getFacts(const CFGBlock *B) const {
    auto It = BlockToFactsMap.find(B);
    if (It != BlockToFactsMap.end())
      return It->second;
    return {};
  }

  void addBlockFacts(const CFGBlock *B, llvm::ArrayRef<Fact *> NewFacts) {
    if (!NewFacts.empty())
      BlockToFactsMap[B].assign(NewFacts.begin(), NewFacts.end());
  }

  template <typename FactType, typename... Args>
  FactType *createFact(Args &&...args) {
    void *Mem = FactAllocator.Allocate<FactType>();
    return new (Mem) FactType(std::forward<Args>(args)...);
  }

  void dump(const CFG &Cfg, AnalysisDeclContext &AC) const;

  /// Retrieves program points that were specially marked in the source code
  /// for testing.
  ///
  /// The analysis recognizes special function calls of the form
  /// `void("__lifetime_test_point_<name>")` as test points. This method returns
  /// a map from the annotation string (<name>) to the corresponding
  /// `ProgramPoint`. This allows test harnesses to query the analysis state at
  /// user-defined locations in the code.
  /// \note This is intended for testing only.
  llvm::StringMap<ProgramPoint> getTestPoints() const;

  LoanManager &getLoanMgr() { return LoanMgr; }
  const LoanManager &getLoanMgr() const { return LoanMgr; }
  OriginManager &getOriginMgr() { return OriginMgr; }
  const OriginManager &getOriginMgr() const { return OriginMgr; }

private:
  LoanManager LoanMgr;
  OriginManager OriginMgr;
  llvm::DenseMap<const clang::CFGBlock *, llvm::SmallVector<const Fact *>>
      BlockToFactsMap;
  llvm::BumpPtrAllocator FactAllocator;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H
