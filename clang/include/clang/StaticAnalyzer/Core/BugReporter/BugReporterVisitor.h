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

#ifndef LLVM_CLANG_GR_BUGREPORTERVISITOR
#define LLVM_CLANG_GR_BUGREPORTERVISITOR

#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {

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
  virtual BugReporterVisitor *clone() const = 0;

  /// \brief Return a diagnostic piece which should be associated with the
  /// given node.
  ///
  /// The last parameter can be used to register a new visitor with the given
  /// BugReport while processing a node.
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *PrevN,
                                         BugReporterContext &BRC,
                                         BugReport &BR) = 0;

  /// \brief Provide custom definition for the final diagnostic piece on the
  /// path - the piece, which is displayed before the path is expanded.
  ///
  /// If returns NULL the default implementation will be used.
  /// Also note that at most one visitor of a BugReport should generate a
  /// non-NULL end of path diagnostic piece.
  virtual PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                          const ExplodedNode *N,
                                          BugReport &BR);

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;

  /// \brief Generates the default final diagnostic piece.
  static PathDiagnosticPiece *getDefaultEndPath(BugReporterContext &BRC,
                                                const ExplodedNode *N,
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
  virtual BugReporterVisitor *clone() const {
    return new DERIVED(*static_cast<const DERIVED *>(this));
  }
};

class FindLastStoreBRVisitor
  : public BugReporterVisitorImpl<FindLastStoreBRVisitor>
{
  const MemRegion *R;
  SVal V;
  bool satisfied;

public:
  /// \brief Convenience method to create a visitor given only the MemRegion.
  /// Returns NULL if the visitor cannot be created. For example, when the
  /// corresponding value is unknown.
  static BugReporterVisitor *createVisitorObject(const ExplodedNode *N,
                                                 const MemRegion *R);

  /// Creates a visitor for every VarDecl inside a Stmt and registers it with
  /// the BugReport.
  static void registerStatementVarDecls(BugReport &BR, const Stmt *S);

  FindLastStoreBRVisitor(SVal v, const MemRegion *r)
  : R(r), V(v), satisfied(false) {
    assert (!V.isUnknown() && "Cannot track unknown value.");

    // TODO: Does it make sense to allow undef values here?
    // (If not, also see UndefCapturedBlockVarChecker)?
  }

  void Profile(llvm::FoldingSetNodeID &ID) const;

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

class TrackConstraintBRVisitor
  : public BugReporterVisitorImpl<TrackConstraintBRVisitor>
{
  DefinedSVal Constraint;
  const bool Assumption;
  bool isSatisfied;

public:
  TrackConstraintBRVisitor(DefinedSVal constraint, bool assumption)
  : Constraint(constraint), Assumption(assumption), isSatisfied(false) {}

  void Profile(llvm::FoldingSetNodeID &ID) const;

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

class NilReceiverBRVisitor
  : public BugReporterVisitorImpl<NilReceiverBRVisitor>
{
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

/// Visitor that tries to report interesting diagnostics from conditions.
class ConditionBRVisitor : public BugReporterVisitorImpl<ConditionBRVisitor> {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  /// Return the tag associated with this visitor.  This tag will be used
  /// to make all PathDiagnosticPieces created by this visitor.
  static const char *getTag();
  
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *Prev,
                                         BugReporterContext &BRC,
                                         BugReport &BR);

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
                    llvm::raw_ostream &Out,
                    BugReporterContext &BRC,
                    BugReport &R,
                    const ExplodedNode *N,
                    llvm::Optional<bool> &prunable);
};

/// \brief When a region containing undefined value or '0' value is passed 
/// as an argument in a call, marks the call as interesting.
///
/// As a result, BugReporter will not prune the path through the function even
/// if the region's contents are not modified/accessed by the call.
class UndefOrNullArgVisitor
  : public BugReporterVisitorImpl<UndefOrNullArgVisitor> {

  /// The interesting memory region this visitor is tracking.
  const MemRegion *R;

public:
  UndefOrNullArgVisitor(const MemRegion *InR) : R(InR) {}

  virtual void Profile(llvm::FoldingSetNodeID &ID) const {
    static int Tag = 0;
    ID.AddPointer(&Tag);
    ID.AddPointer(R);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
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
///
/// \return Whether or not the function was able to add visitors for this
///         statement. Note that returning \c true does not actually imply
///         that any visitors were added.
bool trackNullOrUndefValue(const ExplodedNode *N, const Stmt *S, BugReport &R,
                           bool IsArg = false);

const Stmt *GetDerefExpr(const ExplodedNode *N);
const Stmt *GetDenomExpr(const ExplodedNode *N);
const Stmt *GetRetValExpr(const ExplodedNode *N);
bool isDeclRefExprToReference(const Expr *E);


} // end namespace clang
} // end namespace ento
} // end namespace bugreporter


#endif //LLVM_CLANG_GR__BUGREPORTERVISITOR
