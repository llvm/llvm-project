//===---  BugReporterVisitor.h - Generate PathDiagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares BugReporterVisitors, which are used to generate enhanced
//  diagnostic traces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGREPORTERVISITOR_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGREPORTERVISITOR_H

#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {
class CFGBlock;

namespace ento {

class BugReport;
class BugReporterContext;
class ExplodedNode;
class MemRegion;
class PathDiagnosticPiece;

/// \brief BugReporterVisitors are used to add custom diagnostics along a path.
///
/// Custom visitors should subclass the BugReporterVisitorImpl class for a
/// default implementation of the clone() method.
/// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
/// default implementation of clone() will NOT do the right thing, and you
/// will have to provide your own implementation.)
class BugReporterVisitor : public llvm::FoldingSetNode {
public:
  BugReporterVisitor() = default;
  BugReporterVisitor(const BugReporterVisitor &) = default;
  BugReporterVisitor(BugReporterVisitor &&) {}
  virtual ~BugReporterVisitor();

  /// \brief Returns a copy of this BugReporter.
  ///
  /// Custom BugReporterVisitors should not override this method directly.
  /// Instead, they should inherit from BugReporterVisitorImpl and provide
  /// a protected or public copy constructor.
  ///
  /// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
  /// default implementation of clone() will NOT do the right thing, and you
  /// will have to provide your own implementation.)
  virtual std::unique_ptr<BugReporterVisitor> clone() const = 0;

  /// \brief Return a diagnostic piece which should be associated with the
  /// given node.
  ///
  /// The last parameter can be used to register a new visitor with the given
  /// BugReport while processing a node.
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *Succ,
                                         const ExplodedNode *Pred,
                                         BugReporterContext &BRC,
                                         BugReport &BR) = 0;

  /// \brief Provide custom definition for the final diagnostic piece on the
  /// path - the piece, which is displayed before the path is expanded.
  ///
  /// If returns NULL the default implementation will be used.
  /// Also note that at most one visitor of a BugReport should generate a
  /// non-NULL end of path diagnostic piece.
  virtual std::unique_ptr<PathDiagnosticPiece>
  getEndPath(BugReporterContext &BRC, const ExplodedNode *N, BugReport &BR);

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;

  /// \brief Generates the default final diagnostic piece.
  static std::unique_ptr<PathDiagnosticPiece>
  getDefaultEndPath(BugReporterContext &BRC, const ExplodedNode *N,
                    BugReport &BR);
};

/// This class provides a convenience implementation for clone() using the
/// Curiously-Recurring Template Pattern. If you are implementing a custom
/// BugReporterVisitor, subclass BugReporterVisitorImpl and provide a public
/// or protected copy constructor.
///
/// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
/// default implementation of clone() will NOT do the right thing, and you
/// will have to provide your own implementation.)
template <class DERIVED>
class BugReporterVisitorImpl : public BugReporterVisitor {
  std::unique_ptr<BugReporterVisitor> clone() const override {
    return llvm::make_unique<DERIVED>(*static_cast<const DERIVED *>(this));
  }
};

class FindLastStoreBRVisitor final
    : public BugReporterVisitorImpl<FindLastStoreBRVisitor> {
  const MemRegion *R;
  SVal V;
  bool Satisfied;

  /// If the visitor is tracking the value directly responsible for the
  /// bug, we are going to employ false positive suppression.
  bool EnableNullFPSuppression;

public:
  /// Creates a visitor for every VarDecl inside a Stmt and registers it with
  /// the BugReport.
  static void registerStatementVarDecls(BugReport &BR, const Stmt *S,
                                        bool EnableNullFPSuppression);

  FindLastStoreBRVisitor(KnownSVal V, const MemRegion *R,
                         bool InEnableNullFPSuppression)
  : R(R),
    V(V),
    Satisfied(false),
    EnableNullFPSuppression(InEnableNullFPSuppression) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;
};

class TrackConstraintBRVisitor final
    : public BugReporterVisitorImpl<TrackConstraintBRVisitor> {
  DefinedSVal Constraint;
  bool Assumption;
  bool IsSatisfied;
  bool IsZeroCheck;

  /// We should start tracking from the last node along the path in which the
  /// value is constrained.
  bool IsTrackingTurnedOn;

public:
  TrackConstraintBRVisitor(DefinedSVal constraint, bool assumption)
  : Constraint(constraint), Assumption(assumption), IsSatisfied(false),
    IsZeroCheck(!Assumption && Constraint.getAs<Loc>()),
    IsTrackingTurnedOn(false) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;

private:
  /// Checks if the constraint is valid in the current state.
  bool isUnderconstrained(const ExplodedNode *N) const;

};

/// \class NilReceiverBRVisitor
/// \brief Prints path notes when a message is sent to a nil receiver.
class NilReceiverBRVisitor final
    : public BugReporterVisitorImpl<NilReceiverBRVisitor> {
public:

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;

  /// If the statement is a message send expression with nil receiver, returns
  /// the receiver expression. Returns NULL otherwise.
  static const Expr *getNilReceiver(const Stmt *S, const ExplodedNode *N);
};

/// Visitor that tries to report interesting diagnostics from conditions.
class ConditionBRVisitor final
    : public BugReporterVisitorImpl<ConditionBRVisitor> {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
  }

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *Prev,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;

  PathDiagnosticPiece *VisitNodeImpl(const ExplodedNode *N,
                                     const ExplodedNode *Prev,
                                     BugReporterContext &BRC,
                                     BugReport &BR);
  
  PathDiagnosticPiece *VisitTerminator(const Stmt *Term,
                                       const ExplodedNode *N,
                                       const CFGBlock *srcBlk,
                                       const CFGBlock *dstBlk,
                                       BugReport &R,
                                       BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const DeclRefExpr *DR,
                                     const bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const BinaryOperator *BExpr,
                                     const bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);
  
  PathDiagnosticPiece *VisitConditionVariable(StringRef LhsString,
                                              const Expr *CondVarExpr,
                                              const bool tookTrue,
                                              BugReporterContext &BRC,
                                              BugReport &R,
                                              const ExplodedNode *N);

  bool patternMatch(const Expr *Ex,
                    raw_ostream &Out,
                    BugReporterContext &BRC,
                    BugReport &R,
                    const ExplodedNode *N,
                    Optional<bool> &prunable);
};

/// \brief Suppress reports that might lead to known false positives.
///
/// Currently this suppresses reports based on locations of bugs.
class LikelyFalsePositiveSuppressionBRVisitor final
    : public BugReporterVisitorImpl<LikelyFalsePositiveSuppressionBRVisitor> {
public:
  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(getTag());
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *Prev,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override {
    return nullptr;
  }

  std::unique_ptr<PathDiagnosticPiece> getEndPath(BugReporterContext &BRC,
                                                  const ExplodedNode *N,
                                                  BugReport &BR) override;
};

/// \brief When a region containing undefined value or '0' value is passed 
/// as an argument in a call, marks the call as interesting.
///
/// As a result, BugReporter will not prune the path through the function even
/// if the region's contents are not modified/accessed by the call.
class UndefOrNullArgVisitor final
    : public BugReporterVisitorImpl<UndefOrNullArgVisitor> {

  /// The interesting memory region this visitor is tracking.
  const MemRegion *R;

public:
  UndefOrNullArgVisitor(const MemRegion *InR) : R(InR) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int Tag = 0;
    ID.AddPointer(&Tag);
    ID.AddPointer(R);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;
};

class SuppressInlineDefensiveChecksVisitor final
    : public BugReporterVisitorImpl<SuppressInlineDefensiveChecksVisitor> {
  /// The symbolic value for which we are tracking constraints.
  /// This value is constrained to null in the end of path.
  DefinedSVal V;

  /// Track if we found the node where the constraint was first added.
  bool IsSatisfied;

  /// Since the visitors can be registered on nodes previous to the last
  /// node in the BugReport, but the path traversal always starts with the last
  /// node, the visitor invariant (that we start with a node in which V is null)
  /// might not hold when node visitation starts. We are going to start tracking
  /// from the last node in which the value is null.
  bool IsTrackingTurnedOn;

public:
  SuppressInlineDefensiveChecksVisitor(DefinedSVal Val, const ExplodedNode *N);

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPiece *VisitNode(const ExplodedNode *Succ,
                                 const ExplodedNode *Pred,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;
};

class CXXSelfAssignmentBRVisitor final
  : public BugReporterVisitorImpl<CXXSelfAssignmentBRVisitor> {
  
  bool Satisfied;

public:
  CXXSelfAssignmentBRVisitor() : Satisfied(false) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {}

  PathDiagnosticPiece *VisitNode(const ExplodedNode *Succ,
                                 const ExplodedNode *Pred,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;
};

namespace bugreporter {

/// Attempts to add visitors to trace a null or undefined value back to its
/// point of origin, whether it is a symbol constrained to null or an explicit
/// assignment.
///
/// \param N A node "downstream" from the evaluation of the statement.
/// \param S The statement whose value is null or undefined.
/// \param R The bug report to which visitors should be attached.
/// \param IsArg Whether the statement is an argument to an inlined function.
///              If this is the case, \p N \em must be the CallEnter node for
///              the function.
/// \param EnableNullFPSuppression Whether we should employ false positive
///         suppression (inlined defensive checks, returned null).
///
/// \return Whether or not the function was able to add visitors for this
///         statement. Note that returning \c true does not actually imply
///         that any visitors were added.
bool trackNullOrUndefValue(const ExplodedNode *N, const Stmt *S, BugReport &R,
                           bool IsArg = false,
                           bool EnableNullFPSuppression = true);

const Expr *getDerefExpr(const Stmt *S);
const Stmt *GetDenomExpr(const ExplodedNode *N);
const Stmt *GetRetValExpr(const ExplodedNode *N);
bool isDeclRefExprToReference(const Expr *E);


} // end namespace clang
} // end namespace ento
} // end namespace bugreporter


#endif
