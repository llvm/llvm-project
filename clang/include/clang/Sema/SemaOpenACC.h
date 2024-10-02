//===----- SemaOpenACC.h - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAOPENACC_H
#define LLVM_CLANG_SEMA_SEMAOPENACC_H

#include "clang/AST/DeclGroup.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <optional>
#include <utility>
#include <variant>

namespace clang {
class IdentifierInfo;
class OpenACCClause;

class SemaOpenACC : public SemaBase {
private:
  /// A collection of loop constructs in the compute construct scope that
  /// haven't had their 'parent' compute construct set yet. Entires will only be
  /// made to this list in the case where we know the loop isn't an orphan.
  llvm::SmallVector<OpenACCLoopConstruct *> ParentlessLoopConstructs;
  /// Whether we are inside of a compute construct, and should add loops to the
  /// above collection.
  bool InsideComputeConstruct = false;

  /// The 'collapse' clause requires quite a bit of checking while
  /// parsing/instantiating its body, so this structure/object keeps all of the
  /// necessary information as we do checking.  This should rarely be directly
  /// modified, and typically should be controlled by the RAII objects.
  ///
  /// Collapse has an 'N' count that makes it apply to a number of loops 'below'
  /// it.
  struct CollapseCheckingInfo {
    OpenACCCollapseClause *ActiveCollapse = nullptr;

    /// This is a value that maintains the current value of the 'N' on the
    /// current collapse, minus the depth that has already been traversed. When
    /// there is not an active collapse, or a collapse whose depth we don't know
    /// (for example, if it is a dependent value), this should be `nullopt`,
    /// else it should be 'N' minus the current depth traversed.
    std::optional<llvm::APSInt> CurCollapseCount;

    /// Records whether we've seen the top level 'for'. We already diagnose
    /// later that the 'top level' is a for loop, so we use this to suppress the
    /// 'collapse inner loop not a 'for' loop' diagnostic.
    LLVM_PREFERRED_TYPE(bool)
    unsigned TopLevelLoopSeen : 1;

    /// Records whether this 'tier' of the loop has already seen a 'for' loop,
    /// used to diagnose if there are multiple 'for' loops at any one level.
    LLVM_PREFERRED_TYPE(bool)
    unsigned CurLevelHasLoopAlready : 1;

    /// Records whether we've hit a CurCollapseCount of '0' on the way down,
    /// which allows us to diagnose if the value of 'N' is too large for the
    /// current number of 'for' loops.
    LLVM_PREFERRED_TYPE(bool)
    unsigned CollapseDepthSatisfied : 1;
  } CollapseInfo{nullptr, std::nullopt, /*TopLevelLoopSeen=*/false,
                 /*CurLevelHasLoopAlready=*/false,
                 /*CollapseDepthSatisfied=*/true};

public:
  // Redeclaration of the version in OpenACCClause.h.
  using DeviceTypeArgument = std::pair<IdentifierInfo *, SourceLocation>;

  /// A type to represent all the data for an OpenACC Clause that has been
  /// parsed, but not yet created/semantically analyzed. This is effectively a
  /// discriminated union on the 'Clause Kind', with all of the individual
  /// clause details stored in a std::variant.
  class OpenACCParsedClause {
    OpenACCDirectiveKind DirKind;
    OpenACCClauseKind ClauseKind;
    SourceRange ClauseRange;
    SourceLocation LParenLoc;

    struct DefaultDetails {
      OpenACCDefaultClauseKind DefaultClauseKind;
    };

    struct ConditionDetails {
      Expr *ConditionExpr;
    };

    struct IntExprDetails {
      SmallVector<Expr *> IntExprs;
    };

    struct VarListDetails {
      SmallVector<Expr *> VarList;
      bool IsReadOnly;
      bool IsZero;
    };

    struct WaitDetails {
      Expr *DevNumExpr;
      SourceLocation QueuesLoc;
      SmallVector<Expr *> QueueIdExprs;
    };

    struct DeviceTypeDetails {
      SmallVector<DeviceTypeArgument> Archs;
    };
    struct ReductionDetails {
      OpenACCReductionOperator Op;
      SmallVector<Expr *> VarList;
    };

    struct CollapseDetails {
      bool IsForce;
      Expr *LoopCount;
    };

    std::variant<std::monostate, DefaultDetails, ConditionDetails,
                 IntExprDetails, VarListDetails, WaitDetails, DeviceTypeDetails,
                 ReductionDetails, CollapseDetails>
        Details = std::monostate{};

  public:
    OpenACCParsedClause(OpenACCDirectiveKind DirKind,
                        OpenACCClauseKind ClauseKind, SourceLocation BeginLoc)
        : DirKind(DirKind), ClauseKind(ClauseKind), ClauseRange(BeginLoc, {}) {}

    OpenACCDirectiveKind getDirectiveKind() const { return DirKind; }

    OpenACCClauseKind getClauseKind() const { return ClauseKind; }

    SourceLocation getBeginLoc() const { return ClauseRange.getBegin(); }

    SourceLocation getLParenLoc() const { return LParenLoc; }

    SourceLocation getEndLoc() const { return ClauseRange.getEnd(); }

    OpenACCDefaultClauseKind getDefaultClauseKind() const {
      assert(ClauseKind == OpenACCClauseKind::Default &&
             "Parsed clause is not a default clause");
      return std::get<DefaultDetails>(Details).DefaultClauseKind;
    }

    const Expr *getConditionExpr() const {
      return const_cast<OpenACCParsedClause *>(this)->getConditionExpr();
    }

    Expr *getConditionExpr() {
      assert((ClauseKind == OpenACCClauseKind::If ||
              (ClauseKind == OpenACCClauseKind::Self &&
               DirKind != OpenACCDirectiveKind::Update)) &&
             "Parsed clause kind does not have a condition expr");

      // 'self' has an optional ConditionExpr, so be tolerant of that. This will
      // assert in variant otherwise.
      if (ClauseKind == OpenACCClauseKind::Self &&
          std::holds_alternative<std::monostate>(Details))
        return nullptr;

      return std::get<ConditionDetails>(Details).ConditionExpr;
    }

    unsigned getNumIntExprs() const {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::Async ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");

      // 'async' and 'wait' have an optional IntExpr, so be tolerant of that.
      if ((ClauseKind == OpenACCClauseKind::Async ||
           ClauseKind == OpenACCClauseKind::Wait) &&
          std::holds_alternative<std::monostate>(Details))
        return 0;
      return std::get<IntExprDetails>(Details).IntExprs.size();
    }

    SourceLocation getQueuesLoc() const {
      assert(ClauseKind == OpenACCClauseKind::Wait &&
             "Parsed clause kind does not have a queues location");

      if (std::holds_alternative<std::monostate>(Details))
        return SourceLocation{};

      return std::get<WaitDetails>(Details).QueuesLoc;
    }

    Expr *getDevNumExpr() const {
      assert(ClauseKind == OpenACCClauseKind::Wait &&
             "Parsed clause kind does not have a device number expr");

      if (std::holds_alternative<std::monostate>(Details))
        return nullptr;

      return std::get<WaitDetails>(Details).DevNumExpr;
    }

    ArrayRef<Expr *> getQueueIdExprs() const {
      assert(ClauseKind == OpenACCClauseKind::Wait &&
             "Parsed clause kind does not have a queue id expr list");

      if (std::holds_alternative<std::monostate>(Details))
        return ArrayRef<Expr *>{std::nullopt};

      return std::get<WaitDetails>(Details).QueueIdExprs;
    }

    ArrayRef<Expr *> getIntExprs() {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::Async ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");

      return std::get<IntExprDetails>(Details).IntExprs;
    }

    ArrayRef<Expr *> getIntExprs() const {
      return const_cast<OpenACCParsedClause *>(this)->getIntExprs();
    }

    OpenACCReductionOperator getReductionOp() const {
      return std::get<ReductionDetails>(Details).Op;
    }

    ArrayRef<Expr *> getVarList() {
      assert((ClauseKind == OpenACCClauseKind::Private ||
              ClauseKind == OpenACCClauseKind::NoCreate ||
              ClauseKind == OpenACCClauseKind::Present ||
              ClauseKind == OpenACCClauseKind::Copy ||
              ClauseKind == OpenACCClauseKind::PCopy ||
              ClauseKind == OpenACCClauseKind::PresentOrCopy ||
              ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn ||
              ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate ||
              ClauseKind == OpenACCClauseKind::Attach ||
              ClauseKind == OpenACCClauseKind::DevicePtr ||
              ClauseKind == OpenACCClauseKind::Reduction ||
              ClauseKind == OpenACCClauseKind::FirstPrivate) &&
             "Parsed clause kind does not have a var-list");

      if (ClauseKind == OpenACCClauseKind::Reduction)
        return std::get<ReductionDetails>(Details).VarList;

      return std::get<VarListDetails>(Details).VarList;
    }

    ArrayRef<Expr *> getVarList() const {
      return const_cast<OpenACCParsedClause *>(this)->getVarList();
    }

    bool isReadOnly() const {
      assert((ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn) &&
             "Only copyin accepts 'readonly:' tag");
      return std::get<VarListDetails>(Details).IsReadOnly;
    }

    bool isZero() const {
      assert((ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate) &&
             "Only copyout/create accepts 'zero' tag");
      return std::get<VarListDetails>(Details).IsZero;
    }

    bool isForce() const {
      assert(ClauseKind == OpenACCClauseKind::Collapse &&
             "Only 'collapse' has a force tag");
      return std::get<CollapseDetails>(Details).IsForce;
    }

    Expr *getLoopCount() const {
      assert(ClauseKind == OpenACCClauseKind::Collapse &&
             "Only 'collapse' has a loop count");
      return std::get<CollapseDetails>(Details).LoopCount;
    }

    ArrayRef<DeviceTypeArgument> getDeviceTypeArchitectures() const {
      assert((ClauseKind == OpenACCClauseKind::DeviceType ||
              ClauseKind == OpenACCClauseKind::DType) &&
             "Only 'device_type'/'dtype' has a device-type-arg list");
      return std::get<DeviceTypeDetails>(Details).Archs;
    }

    void setLParenLoc(SourceLocation EndLoc) { LParenLoc = EndLoc; }
    void setEndLoc(SourceLocation EndLoc) { ClauseRange.setEnd(EndLoc); }

    void setDefaultDetails(OpenACCDefaultClauseKind DefKind) {
      assert(ClauseKind == OpenACCClauseKind::Default &&
             "Parsed clause is not a default clause");
      Details = DefaultDetails{DefKind};
    }

    void setConditionDetails(Expr *ConditionExpr) {
      assert((ClauseKind == OpenACCClauseKind::If ||
              (ClauseKind == OpenACCClauseKind::Self &&
               DirKind != OpenACCDirectiveKind::Update)) &&
             "Parsed clause kind does not have a condition expr");
      // In C++ we can count on this being a 'bool', but in C this gets left as
      // some sort of scalar that codegen will have to take care of converting.
      assert((!ConditionExpr || ConditionExpr->isInstantiationDependent() ||
              ConditionExpr->getType()->isScalarType()) &&
             "Condition expression type not scalar/dependent");

      Details = ConditionDetails{ConditionExpr};
    }

    void setIntExprDetails(ArrayRef<Expr *> IntExprs) {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::Async ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      Details = IntExprDetails{{IntExprs.begin(), IntExprs.end()}};
    }
    void setIntExprDetails(llvm::SmallVector<Expr *> &&IntExprs) {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::Async ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      Details = IntExprDetails{std::move(IntExprs)};
    }

    void setVarListDetails(ArrayRef<Expr *> VarList, bool IsReadOnly,
                           bool IsZero) {
      assert((ClauseKind == OpenACCClauseKind::Private ||
              ClauseKind == OpenACCClauseKind::NoCreate ||
              ClauseKind == OpenACCClauseKind::Present ||
              ClauseKind == OpenACCClauseKind::Copy ||
              ClauseKind == OpenACCClauseKind::PCopy ||
              ClauseKind == OpenACCClauseKind::PresentOrCopy ||
              ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn ||
              ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate ||
              ClauseKind == OpenACCClauseKind::Attach ||
              ClauseKind == OpenACCClauseKind::DevicePtr ||
              ClauseKind == OpenACCClauseKind::FirstPrivate) &&
             "Parsed clause kind does not have a var-list");
      assert((!IsReadOnly || ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn) &&
             "readonly: tag only valid on copyin");
      assert((!IsZero || ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate) &&
             "zero: tag only valid on copyout/create");
      Details =
          VarListDetails{{VarList.begin(), VarList.end()}, IsReadOnly, IsZero};
    }

    void setVarListDetails(llvm::SmallVector<Expr *> &&VarList, bool IsReadOnly,
                           bool IsZero) {
      assert((ClauseKind == OpenACCClauseKind::Private ||
              ClauseKind == OpenACCClauseKind::NoCreate ||
              ClauseKind == OpenACCClauseKind::Present ||
              ClauseKind == OpenACCClauseKind::Copy ||
              ClauseKind == OpenACCClauseKind::PCopy ||
              ClauseKind == OpenACCClauseKind::PresentOrCopy ||
              ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn ||
              ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate ||
              ClauseKind == OpenACCClauseKind::Attach ||
              ClauseKind == OpenACCClauseKind::DevicePtr ||
              ClauseKind == OpenACCClauseKind::FirstPrivate) &&
             "Parsed clause kind does not have a var-list");
      assert((!IsReadOnly || ClauseKind == OpenACCClauseKind::CopyIn ||
              ClauseKind == OpenACCClauseKind::PCopyIn ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyIn) &&
             "readonly: tag only valid on copyin");
      assert((!IsZero || ClauseKind == OpenACCClauseKind::CopyOut ||
              ClauseKind == OpenACCClauseKind::PCopyOut ||
              ClauseKind == OpenACCClauseKind::PresentOrCopyOut ||
              ClauseKind == OpenACCClauseKind::Create ||
              ClauseKind == OpenACCClauseKind::PCreate ||
              ClauseKind == OpenACCClauseKind::PresentOrCreate) &&
             "zero: tag only valid on copyout/create");
      Details = VarListDetails{std::move(VarList), IsReadOnly, IsZero};
    }

    void setReductionDetails(OpenACCReductionOperator Op,
                             llvm::SmallVector<Expr *> &&VarList) {
      assert(ClauseKind == OpenACCClauseKind::Reduction &&
             "reduction details only valid on reduction");
      Details = ReductionDetails{Op, std::move(VarList)};
    }

    void setWaitDetails(Expr *DevNum, SourceLocation QueuesLoc,
                        llvm::SmallVector<Expr *> &&IntExprs) {
      assert(ClauseKind == OpenACCClauseKind::Wait &&
             "Parsed clause kind does not have a wait-details");
      Details = WaitDetails{DevNum, QueuesLoc, std::move(IntExprs)};
    }

    void setDeviceTypeDetails(llvm::SmallVector<DeviceTypeArgument> &&Archs) {
      assert((ClauseKind == OpenACCClauseKind::DeviceType ||
              ClauseKind == OpenACCClauseKind::DType) &&
             "Only 'device_type'/'dtype' has a device-type-arg list");
      Details = DeviceTypeDetails{std::move(Archs)};
    }

    void setCollapseDetails(bool IsForce, Expr *LoopCount) {
      assert(ClauseKind == OpenACCClauseKind::Collapse &&
             "Only 'collapse' has collapse details");
      Details = CollapseDetails{IsForce, LoopCount};
    }
  };

  SemaOpenACC(Sema &S);

  // Called when we encounter a 'while' statement, before looking at its 'body'.
  void ActOnWhileStmt(SourceLocation WhileLoc);
  // Called when we encounter a 'do' statement, before looking at its 'body'.
  void ActOnDoStmt(SourceLocation DoLoc);
  // Called when we encounter a 'for' statement, before looking at its 'body'.
  void ActOnForStmtBegin(SourceLocation ForLoc);
  // Called when we encounter a 'for' statement, after we've consumed/checked
  // the body. This is necessary for a number of checks on the contents of the
  // 'for' statement.
  void ActOnForStmtEnd(SourceLocation ForLoc, StmtResult Body);

  /// Called after parsing an OpenACC Clause so that it can be checked.
  OpenACCClause *ActOnClause(ArrayRef<const OpenACCClause *> ExistingClauses,
                             OpenACCParsedClause &Clause);

  /// Called after the construct has been parsed, but clauses haven't been
  /// parsed.  This allows us to diagnose not-implemented, as well as set up any
  /// state required for parsing the clauses.
  void ActOnConstruct(OpenACCDirectiveKind K, SourceLocation DirLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'statement' context.
  bool ActOnStartStmtDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'Decl' context.
  bool ActOnStartDeclDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);
  /// Called when we encounter an associated statement for our construct, this
  /// should check legality of the statement as it appertains to this Construct.
  StmtResult ActOnAssociatedStmt(SourceLocation DirectiveLoc,
                                 OpenACCDirectiveKind K, StmtResult AssocStmt);

  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  StmtResult ActOnEndStmtDirective(OpenACCDirectiveKind K,
                                   SourceLocation StartLoc,
                                   SourceLocation DirLoc,
                                   SourceLocation EndLoc,
                                   ArrayRef<OpenACCClause *> Clauses,
                                   StmtResult AssocStmt);

  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  DeclGroupRef ActOnEndDeclDirective();

  /// Called when encountering an 'int-expr' for OpenACC, and manages
  /// conversions and diagnostics to 'int'.
  ExprResult ActOnIntExpr(OpenACCDirectiveKind DK, OpenACCClauseKind CK,
                          SourceLocation Loc, Expr *IntExpr);

  /// Called when encountering a 'var' for OpenACC, ensures it is actually a
  /// declaration reference to a variable of the correct type.
  ExprResult ActOnVar(OpenACCClauseKind CK, Expr *VarExpr);

  /// Called while semantically analyzing the reduction clause, ensuring the var
  /// is the correct kind of reference.
  ExprResult CheckReductionVar(Expr *VarExpr);

  /// Called to check the 'var' type is a variable of pointer type, necessary
  /// for 'deviceptr' and 'attach' clauses. Returns true on success.
  bool CheckVarIsPointerType(OpenACCClauseKind ClauseKind, Expr *VarExpr);

  /// Checks and creates an Array Section used in an OpenACC construct/clause.
  ExprResult ActOnArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                   Expr *LowerBound,
                                   SourceLocation ColonLocFirst, Expr *Length,
                                   SourceLocation RBLoc);
  /// Checks the loop depth value for a collapse clause.
  ExprResult CheckCollapseLoopCount(Expr *LoopCount);

  /// Helper type to restore the state of various 'loop' constructs when we run
  /// into a loop (for, etc) inside the construct.
  class LoopInConstructRAII {
    SemaOpenACC &SemaRef;
    CollapseCheckingInfo OldCollapseInfo;
    bool PreserveDepth;

  public:
    LoopInConstructRAII(SemaOpenACC &SemaRef, bool PreserveDepth = true)
        : SemaRef(SemaRef), OldCollapseInfo(SemaRef.CollapseInfo),
          PreserveDepth(PreserveDepth) {}
    ~LoopInConstructRAII() {
      // The associated-statement level of this should NOT preserve this, as it
      // is a new construct, but other loop uses need to preserve the depth so
      // it makes it to the 'top level' for diagnostics.
      bool CollapseDepthSatisified =
          PreserveDepth ? SemaRef.CollapseInfo.CollapseDepthSatisfied
                        : OldCollapseInfo.CollapseDepthSatisfied;
      bool CurLevelHasLoopAlready =
          PreserveDepth ? SemaRef.CollapseInfo.CurLevelHasLoopAlready
                        : OldCollapseInfo.CurLevelHasLoopAlready;
      SemaRef.CollapseInfo = OldCollapseInfo;
      SemaRef.CollapseInfo.CollapseDepthSatisfied = CollapseDepthSatisified;
      SemaRef.CollapseInfo.CurLevelHasLoopAlready = CurLevelHasLoopAlready;
    }
  };

  /// Helper type for the registration/assignment of constructs that need to
  /// 'know' about their parent constructs and hold a reference to them, such as
  /// Loop needing its parent construct.
  class AssociatedStmtRAII {
    SemaOpenACC &SemaRef;
    bool WasInsideComputeConstruct;
    OpenACCDirectiveKind DirKind;
    llvm::SmallVector<OpenACCLoopConstruct *> ParentlessLoopConstructs;
    LoopInConstructRAII LoopRAII;

  public:
    AssociatedStmtRAII(SemaOpenACC &, OpenACCDirectiveKind,
                       ArrayRef<const OpenACCClause *>,
                       ArrayRef<OpenACCClause *>);
    void SetCollapseInfoBeforeAssociatedStmt(
        ArrayRef<const OpenACCClause *> UnInstClauses,
        ArrayRef<OpenACCClause *> Clauses);
    ~AssociatedStmtRAII();
  };
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAOPENACC_H
