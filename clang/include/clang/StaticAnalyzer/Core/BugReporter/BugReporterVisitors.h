//===- BugReporterVisitors.h - Generate PathDiagnostics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares BugReporterVisitors, which are used to generate enhanced
//  diagnostic traces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGREPORTERVISITORS_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGREPORTERVISITORS_H

#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {

class BinaryOperator;
class CFGBlock;
class DeclRefExpr;
class Expr;
class Stmt;

namespace ento {

class PathSensitiveBugReport;
class BugReporterContext;
class ExplodedNode;
class MemRegion;
class PathDiagnosticPiece;
using PathDiagnosticPieceRef = std::shared_ptr<PathDiagnosticPiece>;

/// BugReporterVisitors are used to add custom diagnostics along a path.
class BugReporterVisitor : public llvm::FoldingSetNode {
public:
  BugReporterVisitor() = default;
  BugReporterVisitor(const BugReporterVisitor &) = default;
  BugReporterVisitor(BugReporterVisitor &&) {}
  virtual ~BugReporterVisitor();

  /// Return a diagnostic piece which should be associated with the
  /// given node.
  /// Note that this function does *not* get run on the very last node
  /// of the report, as the PathDiagnosticPiece associated with the
  /// last node should be unique.
  /// Use \ref getEndPath to customize the note associated with the report
  /// end instead.
  ///
  /// The last parameter can be used to register a new visitor with the given
  /// BugReport while processing a node.
  virtual PathDiagnosticPieceRef VisitNode(const ExplodedNode *Succ,
                                           BugReporterContext &BRC,
                                           PathSensitiveBugReport &BR) = 0;

  /// Last function called on the visitor, no further calls to VisitNode
  /// would follow.
  virtual void finalizeVisitor(BugReporterContext &BRC,
                               const ExplodedNode *EndPathNode,
                               PathSensitiveBugReport &BR);

  /// Provide custom definition for the final diagnostic piece on the
  /// path - the piece, which is displayed before the path is expanded.
  ///
  /// NOTE that this function can be implemented on at most one used visitor,
  /// and otherwise it crahes at runtime.
  virtual PathDiagnosticPieceRef getEndPath(BugReporterContext &BRC,
                                            const ExplodedNode *N,
                                            PathSensitiveBugReport &BR);

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;

  /// Generates the default final diagnostic piece.
  static PathDiagnosticPieceRef
  getDefaultEndPath(const BugReporterContext &BRC, const ExplodedNode *N,
                    const PathSensitiveBugReport &BR);
};

namespace bugreporter {

/// Specifies the type of tracking for an expression.
enum class TrackingKind {
  /// Default tracking kind -- specifies that as much information should be
  /// gathered about the tracked expression value as possible.
  Thorough,
  /// Specifies that a more moderate tracking should be used for the expression
  /// value. This will essentially make sure that functions relevant to it
  /// aren't pruned, but otherwise relies on the user reading the code or
  /// following the arrows.
  Condition
};

/// Attempts to add visitors to track expression value back to its point of
/// origin.
///
/// \param N A node "downstream" from the evaluation of the statement.
/// \param E The expression value which we are tracking
/// \param R The bug report to which visitors should be attached.
/// \param EnableNullFPSuppression Whether we should employ false positive
///         suppression (inlined defensive checks, returned null).
///
/// \return Whether or not the function was able to add visitors for this
///         statement. Note that returning \c true does not actually imply
///         that any visitors were added.
bool trackExpressionValue(const ExplodedNode *N, const Expr *E,
                          PathSensitiveBugReport &R,
                          TrackingKind TKind = TrackingKind::Thorough,
                          bool EnableNullFPSuppression = true);

const Expr *getDerefExpr(const Stmt *S);

} // namespace bugreporter

/// Finds last store into the given region,
/// which is different from a given symbolic value.
class FindLastStoreBRVisitor final : public BugReporterVisitor {
  const MemRegion *R;
  SVal V;
  bool Satisfied = false;

  /// If the visitor is tracking the value directly responsible for the
  /// bug, we are going to employ false positive suppression.
  bool EnableNullFPSuppression;

  using TrackingKind = bugreporter::TrackingKind;
  TrackingKind TKind;
  const StackFrameContext *OriginSFC;

public:
  /// \param V We're searching for the store where \c R received this value.
  /// \param R The region we're tracking.
  /// \param TKind May limit the amount of notes added to the bug report.
  /// \param OriginSFC Only adds notes when the last store happened in a
  ///        different stackframe to this one. Disregarded if the tracking kind
  ///        is thorough.
  ///        This is useful, because for non-tracked regions, notes about
  ///        changes to its value in a nested stackframe could be pruned, and
  ///        this visitor can prevent that without polluting the bugpath too
  ///        much.
  FindLastStoreBRVisitor(KnownSVal V, const MemRegion *R,
                         bool InEnableNullFPSuppression, TrackingKind TKind,
                         const StackFrameContext *OriginSFC = nullptr)
      : R(R), V(V), EnableNullFPSuppression(InEnableNullFPSuppression),
        TKind(TKind), OriginSFC(OriginSFC) {
    assert(R);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};

class TrackConstraintBRVisitor final : public BugReporterVisitor {
  DefinedSVal Constraint;
  bool Assumption;
  bool IsSatisfied = false;
  bool IsZeroCheck;

  /// We should start tracking from the last node along the path in which the
  /// value is constrained.
  bool IsTrackingTurnedOn = false;

public:
  TrackConstraintBRVisitor(DefinedSVal constraint, bool assumption)
      : Constraint(constraint), Assumption(assumption),
        IsZeroCheck(!Assumption && Constraint.getAs<Loc>()) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;

private:
  /// Checks if the constraint is valid in the current state.
  bool isUnderconstrained(const ExplodedNode *N) const;
};

/// \class NilReceiverBRVisitor
/// Prints path notes when a message is sent to a nil receiver.
class NilReceiverBRVisitor final : public BugReporterVisitor {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;

  /// If the statement is a message send expression with nil receiver, returns
  /// the receiver expression. Returns NULL otherwise.
  static const Expr *getNilReceiver(const Stmt *S, const ExplodedNode *N);
};

/// Visitor that tries to report interesting diagnostics from conditions.
class ConditionBRVisitor final : public BugReporterVisitor {
  // FIXME: constexpr initialization isn't supported by MSVC2013.
  constexpr static llvm::StringLiteral GenericTrueMessage =
      "Assuming the condition is true";
  constexpr static llvm::StringLiteral GenericFalseMessage =
      "Assuming the condition is false";

public:
  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
  }

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;

  PathDiagnosticPieceRef VisitNodeImpl(const ExplodedNode *N,
                                       BugReporterContext &BRC,
                                       PathSensitiveBugReport &BR);

  PathDiagnosticPieceRef
  VisitTerminator(const Stmt *Term, const ExplodedNode *N,
                  const CFGBlock *SrcBlk, const CFGBlock *DstBlk,
                  PathSensitiveBugReport &R, BugReporterContext &BRC);

  PathDiagnosticPieceRef VisitTrueTest(const Expr *Cond,
                                       BugReporterContext &BRC,
                                       PathSensitiveBugReport &R,
                                       const ExplodedNode *N, bool TookTrue);

  PathDiagnosticPieceRef VisitTrueTest(const Expr *Cond, const DeclRefExpr *DR,
                                       BugReporterContext &BRC,
                                       PathSensitiveBugReport &R,
                                       const ExplodedNode *N, bool TookTrue,
                                       bool IsAssuming);

  PathDiagnosticPieceRef
  VisitTrueTest(const Expr *Cond, const BinaryOperator *BExpr,
                BugReporterContext &BRC, PathSensitiveBugReport &R,
                const ExplodedNode *N, bool TookTrue, bool IsAssuming);

  PathDiagnosticPieceRef VisitTrueTest(const Expr *Cond, const MemberExpr *ME,
                                       BugReporterContext &BRC,
                                       PathSensitiveBugReport &R,
                                       const ExplodedNode *N, bool TookTrue,
                                       bool IsAssuming);

  PathDiagnosticPieceRef
  VisitConditionVariable(StringRef LhsString, const Expr *CondVarExpr,
                         BugReporterContext &BRC, PathSensitiveBugReport &R,
                         const ExplodedNode *N, bool TookTrue);

  /// Tries to print the value of the given expression.
  ///
  /// \param CondVarExpr The expression to print its value.
  /// \param Out The stream to print.
  /// \param N The node where we encountered the condition.
  /// \param TookTrue Whether we took the \c true branch of the condition.
  ///
  /// \return Whether the print was successful. (The printing is successful if
  ///         we model the value and we could obtain it.)
  bool printValue(const Expr *CondVarExpr, raw_ostream &Out,
                  const ExplodedNode *N, bool TookTrue, bool IsAssuming);

  bool patternMatch(const Expr *Ex,
                    const Expr *ParentEx,
                    raw_ostream &Out,
                    BugReporterContext &BRC,
                    PathSensitiveBugReport &R,
                    const ExplodedNode *N,
                    Optional<bool> &prunable,
                    bool IsSameFieldName);

  static bool isPieceMessageGeneric(const PathDiagnosticPiece *Piece);
};

/// Suppress reports that might lead to known false positives.
///
/// Currently this suppresses reports based on locations of bugs.
class LikelyFalsePositiveSuppressionBRVisitor final
    : public BugReporterVisitor {
public:
  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(getTag());
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *, BugReporterContext &,
                                   PathSensitiveBugReport &) override {
    return nullptr;
  }

  void finalizeVisitor(BugReporterContext &BRC, const ExplodedNode *N,
                       PathSensitiveBugReport &BR) override;
};

/// When a region containing undefined value or '0' value is passed
/// as an argument in a call, marks the call as interesting.
///
/// As a result, BugReporter will not prune the path through the function even
/// if the region's contents are not modified/accessed by the call.
class UndefOrNullArgVisitor final : public BugReporterVisitor {
  /// The interesting memory region this visitor is tracking.
  const MemRegion *R;

public:
  UndefOrNullArgVisitor(const MemRegion *InR) : R(InR) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int Tag = 0;
    ID.AddPointer(&Tag);
    ID.AddPointer(R);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};

class SuppressInlineDefensiveChecksVisitor final : public BugReporterVisitor {
  /// The symbolic value for which we are tracking constraints.
  /// This value is constrained to null in the end of path.
  DefinedSVal V;

  /// Track if we found the node where the constraint was first added.
  bool IsSatisfied = false;

  /// Since the visitors can be registered on nodes previous to the last
  /// node in the BugReport, but the path traversal always starts with the last
  /// node, the visitor invariant (that we start with a node in which V is null)
  /// might not hold when node visitation starts. We are going to start tracking
  /// from the last node in which the value is null.
  bool IsTrackingTurnedOn = false;

public:
  SuppressInlineDefensiveChecksVisitor(DefinedSVal Val, const ExplodedNode *N);

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *Succ,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};

/// The bug visitor will walk all the nodes in a path and collect all the
/// constraints. When it reaches the root node, will create a refutation
/// manager and check if the constraints are satisfiable
class FalsePositiveRefutationBRVisitor final : public BugReporterVisitor {
private:
  /// Holds the constraints in a given path
  ConstraintMap Constraints;

public:
  FalsePositiveRefutationBRVisitor();

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;

  void finalizeVisitor(BugReporterContext &BRC, const ExplodedNode *EndPathNode,
                       PathSensitiveBugReport &BR) override;
  void addConstraints(const ExplodedNode *N,
                      bool OverwriteConstraintsOnExistingSyms);
};

/// The visitor detects NoteTags and displays the event notes they contain.
class TagVisitor : public BugReporterVisitor {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override;

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &R) override;
};

} // namespace ento

} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGREPORTERVISITORS_H
