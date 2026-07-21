//===- InvalidationCause.h - Cause of a region invalidation ------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines InvalidationCause, a small class hierarchy describing why
// a memory region was invalidated by ProgramState::invalidateRegions. The
// cause is carried by SymbolInvalidationArtifact symbols so that downstream
// machinery (bug-report suppression, diagnostics) can distinguish symbolic
// values produced by an invalidation event from ordinary conjured symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONCAUSE_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONCAUSE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Compiler.h"

namespace clang {

class CallExpr;
class Stmt;

namespace ento {

class SymbolManager;

/// Describes why a memory region was invalidated. Instances are uniqued by
/// SymbolManager::acquireCause<T>(...) and are stable for the analysis
/// lifetime; callers must not allocate them on the stack.
class InvalidationCause : public llvm::FoldingSetNode {
public:
  virtual ~InvalidationCause() = default;

  enum Kind {
    // UnmodeledCall range
    ConservativeEvalCallKind,
    PartiallyModeledCallKind,
    BEGIN_UNMODELED_CALL = ConservativeEvalCallKind,
    END_UNMODELED_CALL = PartiallyModeledCallKind,

    // UnmodeledStmt range
    UnmodeledExprKind,
    LoopWideningKind,
    BEGIN_UNMODELED_STMT = UnmodeledExprKind,
    END_UNMODELED_STMT = LoopWideningKind,
  };

  Kind getKind() const { return K; }

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;
  virtual void dumpToStream(raw_ostream &OS) const = 0;

  LLVM_DUMP_METHOD void dump() const;

protected:
  explicit InvalidationCause(Kind K) : K(K) {}
  virtual void anchor();

private:
  Kind K;
};

/// Abstract base for invalidations triggered by an unmodeled or partially
/// modeled call.
class UnmodeledCall : public InvalidationCause {
public:
  /// Returns the expression whose value will be the result of this call.
  /// Null if and only if 'this' is a CXXDestructorCall.
  const CallExpr *getCallExpr() const { return CE; }

  static bool classof(const InvalidationCause *C) {
    return C->getKind() >= BEGIN_UNMODELED_CALL &&
           C->getKind() <= END_UNMODELED_CALL;
  }

protected:
  const CallExpr *CE;
  UnmodeledCall(Kind K, const CallExpr *CE) : InvalidationCause(K), CE(CE) {}
  void anchor() override;
};

/// Conservative evaluation of a call: the call's body wasn't inlined and we
/// fall back to invalidating its arguments / reachable globals.
class ConservativeEvalCall final : public UnmodeledCall {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override { Profile(ID, CE); }

  static void Profile(llvm::FoldingSetNodeID &ID, const CallExpr *CE) {
    ID.AddInteger((unsigned)ConservativeEvalCallKind);
    ID.AddPointer(CE);
  }

  void dumpToStream(raw_ostream &OS) const override;

  static bool classof(const InvalidationCause *C) {
    return C->getKind() == ConservativeEvalCallKind;
  }

protected:
  friend class SymbolManager;
  explicit ConservativeEvalCall(const CallExpr *CE)
      : UnmodeledCall(ConservativeEvalCallKind, CE) {}
  void anchor() override;
};

/// A call that the analyzer models but bails out of for some operands (e.g.
/// CStringChecker's memcpy fallback, MallocChecker's free invalidation,
/// SmartPtrModeling's ostream<< handling).
class PartiallyModeledCall final : public UnmodeledCall {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override { Profile(ID, CE); }

  static void Profile(llvm::FoldingSetNodeID &ID, const CallExpr *CE) {
    ID.AddInteger((unsigned)PartiallyModeledCallKind);
    ID.AddPointer(CE);
  }

  void dumpToStream(raw_ostream &OS) const override;

  static bool classof(const InvalidationCause *C) {
    return C->getKind() == PartiallyModeledCallKind;
  }

protected:
  friend class SymbolManager;
  explicit PartiallyModeledCall(const CallExpr *CE)
      : UnmodeledCall(PartiallyModeledCallKind, CE) {}
  void anchor() override;
};

/// Abstract base for invalidations triggered by an unmodeled statement
/// (atomics, inline asm) or a widened loop.
class UnmodeledStmt : public InvalidationCause {
public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const Stmt *getStmt() const { return S; }

  static bool classof(const InvalidationCause *C) {
    return C->getKind() >= BEGIN_UNMODELED_STMT &&
           C->getKind() <= END_UNMODELED_STMT;
  }

protected:
  const Stmt *S;
  UnmodeledStmt(Kind K, const Stmt *S) : InvalidationCause(K), S(S) {
    assert(S);
  }
  void anchor() override;
};

/// An expression we don't model (e.g. AtomicExpr, GCCAsmStmt).
class UnmodeledExpr final : public UnmodeledStmt {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override { Profile(ID, S); }

  static void Profile(llvm::FoldingSetNodeID &ID, const Stmt *S) {
    ID.AddInteger((unsigned)UnmodeledExprKind);
    ID.AddPointer(S);
  }

  void dumpToStream(raw_ostream &OS) const override;

  static bool classof(const InvalidationCause *C) {
    return C->getKind() == UnmodeledExprKind;
  }

private:
  friend class SymbolManager;
  explicit UnmodeledExpr(const Stmt *S) : UnmodeledStmt(UnmodeledExprKind, S) {}
  void anchor() override;
};

/// The widened loop's invalidation event.
class LoopWidening final : public UnmodeledStmt {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override { Profile(ID, S); }

  static void Profile(llvm::FoldingSetNodeID &ID, const Stmt *S) {
    ID.AddInteger((unsigned)LoopWideningKind);
    ID.AddPointer(S);
  }

  void dumpToStream(raw_ostream &OS) const override;

  static bool classof(const InvalidationCause *C) {
    return C->getKind() == LoopWideningKind;
  }

protected:
  friend class SymbolManager;
  explicit LoopWidening(const Stmt *S) : UnmodeledStmt(LoopWideningKind, S) {}
  void anchor() override;
};

raw_ostream &operator<<(raw_ostream &OS, const InvalidationCause &C);

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONCAUSE_H
