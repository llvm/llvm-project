//===- StmtOpenMP.h - Classes for OpenMP directives  ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file defines OpenMP AST classes for executable directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTOPENMP_H
#define LLVM_CLANG_AST_STMTOPENMP_H

#include "clang/AST/Expr.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for directives.
//===----------------------------------------------------------------------===//

/// \brief This is a basic class for representing single OpenMP executable
/// directive.
///
class OMPExecutableDirective : public Stmt {
  friend class ASTStmtReader;
  /// \brief Kind of the directive.
  OpenMPDirectiveKind Kind;
  /// \brief Starting location of the directive (directive keyword).
  SourceLocation StartLoc;
  /// \brief Ending location of the directive.
  SourceLocation EndLoc;
  /// \brief Numbers of clauses.
  const unsigned NumClauses;
  /// \brief Number of child expressions/stmts.
  const unsigned NumChildren;
  /// \brief Offset from this to the start of clauses.
  /// There are NumClauses pointers to clauses, they are followed by
  /// NumChildren pointers to child stmts/exprs (if the directive type
  /// requires an associated stmt, then it has to be the first of them).
  const unsigned ClausesOffset;

  /// \brief Get the clauses storage.
  MutableArrayRef<OMPClause *> getClauses() {
    OMPClause **ClauseStorage = reinterpret_cast<OMPClause **>(
        reinterpret_cast<char *>(this) + ClausesOffset);
    return MutableArrayRef<OMPClause *>(ClauseStorage, NumClauses);
  }

protected:
  /// \brief Build instance of directive of class \a K.
  ///
  /// \param SC Statement class.
  /// \param K Kind of OpenMP directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  ///
  template <typename T>
  OMPExecutableDirective(const T *, StmtClass SC, OpenMPDirectiveKind K,
                         SourceLocation StartLoc, SourceLocation EndLoc,
                         unsigned NumClauses, unsigned NumChildren)
      : Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
        EndLoc(std::move(EndLoc)), NumClauses(NumClauses),
        NumChildren(NumChildren),
        ClausesOffset(llvm::alignTo(sizeof(T), llvm::alignOf<OMPClause *>())) {}

  /// \brief Sets the list of variables for this clause.
  ///
  /// \param Clauses The list of clauses for the directive.
  ///
  void setClauses(ArrayRef<OMPClause *> Clauses);

  /// \brief Set the associated statement for the directive.
  ///
  /// /param S Associated statement.
  ///
  void setAssociatedStmt(Stmt *S) {
    assert(hasAssociatedStmt() && "no associated statement.");
    *child_begin() = S;
  }

public:
  /// \brief Iterates over a filtered subrange of clauses applied to a
  /// directive.
  ///
  /// This iterator visits only clauses of type SpecificClause.
  template <typename SpecificClause>
  class specific_clause_iterator
      : public llvm::iterator_adaptor_base<
            specific_clause_iterator<SpecificClause>,
            ArrayRef<OMPClause *>::const_iterator, std::forward_iterator_tag,
            const SpecificClause *, ptrdiff_t, const SpecificClause *,
            const SpecificClause *> {
    ArrayRef<OMPClause *>::const_iterator End;

    void SkipToNextClause() {
      while (this->I != End && !isa<SpecificClause>(*this->I))
        ++this->I;
    }

  public:
    explicit specific_clause_iterator(ArrayRef<OMPClause *> Clauses)
        : specific_clause_iterator::iterator_adaptor_base(Clauses.begin()),
          End(Clauses.end()) {
      SkipToNextClause();
    }

    const SpecificClause *operator*() const {
      return cast<SpecificClause>(*this->I);
    }
    const SpecificClause *operator->() const { return **this; }

    specific_clause_iterator &operator++() {
      ++this->I;
      SkipToNextClause();
      return *this;
    }
  };

  template <typename SpecificClause>
  static llvm::iterator_range<specific_clause_iterator<SpecificClause>>
  getClausesOfKind(ArrayRef<OMPClause *> Clauses) {
    return {specific_clause_iterator<SpecificClause>(Clauses),
            specific_clause_iterator<SpecificClause>(
                llvm::makeArrayRef(Clauses.end(), 0))};
  }

  template <typename SpecificClause>
  llvm::iterator_range<specific_clause_iterator<SpecificClause>>
  getClausesOfKind() const {
    return getClausesOfKind<SpecificClause>(clauses());
  }

  /// Gets a single clause of the specified kind associated with the
  /// current directive iff there is only one clause of this kind (and assertion
  /// is fired if there is more than one clause is associated with the
  /// directive). Returns nullptr if no clause of this kind is associated with
  /// the directive.
  template <typename SpecificClause>
  const SpecificClause *getSingleClause() const {
    auto Clauses = getClausesOfKind<SpecificClause>();

    if (Clauses.begin() != Clauses.end()) {
      assert(std::next(Clauses.begin()) == Clauses.end() &&
             "There are at least 2 clauses of the specified kind");
      return *Clauses.begin();
    }
    return nullptr;
  }

  /// Returns true if the current directive has one or more clauses of a
  /// specific kind.
  template <typename SpecificClause>
  bool hasClausesOfKind() const {
    auto Clauses = getClausesOfKind<SpecificClause>();
    return Clauses.begin() != Clauses.end();
  }

  /// \brief Returns starting location of directive kind.
  SourceLocation getLocStart() const { return StartLoc; }
  /// \brief Returns ending location of directive.
  SourceLocation getLocEnd() const { return EndLoc; }

  /// \brief Set starting location of directive kind.
  ///
  /// \param Loc New starting location of directive.
  ///
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// \brief Set ending location of directive.
  ///
  /// \param Loc New ending location of directive.
  ///
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// \brief Get number of clauses.
  unsigned getNumClauses() const { return NumClauses; }

  /// \brief Returns specified clause.
  ///
  /// \param i Number of clause.
  ///
  OMPClause *getClause(unsigned i) const { return clauses()[i]; }

  /// \brief Returns true if directive has associated statement.
  bool hasAssociatedStmt() const { return NumChildren > 0; }

  /// \brief Returns statement associated with the directive.
  Stmt *getAssociatedStmt() const {
    assert(hasAssociatedStmt() && "no associated statement.");
    return const_cast<Stmt *>(*child_begin());
  }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOMPExecutableDirectiveConstant &&
           S->getStmtClass() <= lastOMPExecutableDirectiveConstant;
  }

  child_range children() {
    if (!hasAssociatedStmt())
      return child_range(child_iterator(), child_iterator());
    Stmt **ChildStorage = reinterpret_cast<Stmt **>(getClauses().end());
    return child_range(ChildStorage, ChildStorage + NumChildren);
  }

  ArrayRef<OMPClause *> clauses() { return getClauses(); }

  ArrayRef<OMPClause *> clauses() const {
    return const_cast<OMPExecutableDirective *>(this)->getClauses();
  }
};

/// \brief This represents '#pragma omp parallel' directive.
///
/// \code
/// #pragma omp parallel private(a,b) reduction(+: c,d)
/// \endcode
/// In this example directive '#pragma omp parallel' has clauses 'private'
/// with the variables 'a' and 'b' and 'reduction' with operator '+' and
/// variables 'c' and 'd'.
///
class OMPParallelDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief true if the construct has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending Location of the directive.
  ///
  OMPParallelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                               StartLoc, EndLoc, NumClauses, 1),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement associated with the directive.
  /// \param HasCancel true if this directive has inner cancel directive.
  ///
  static OMPParallelDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelDirectiveClass;
  }
};

/// \brief This is a common base class for loop directives ('omp simd', 'omp
/// for', 'omp for simd' etc.). It is responsible for the loop code generation.
///
class OMPLoopDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Number of collapsed loops as specified by 'collapse' clause.
  unsigned CollapsedNum;

  /// \brief Offsets to the stored exprs.
  /// This enumeration contains offsets to all the pointers to children
  /// expressions stored in OMPLoopDirective.
  /// The first 9 children are nesessary for all the loop directives, and
  /// the next 7 are specific to the worksharing ones.
  /// After the fixed children, three arrays of length CollapsedNum are
  /// allocated: loop counters, their updates and final values.
  ///
  enum {
    AssociatedStmtOffset = 0,
    IterationVariableOffset = 1,
    LastIterationOffset = 2,
    CalcLastIterationOffset = 3,
    PreConditionOffset = 4,
    CondOffset = 5,
    InitOffset = 6,
    IncOffset = 7,
    // The '...End' enumerators do not correspond to child expressions - they
    // specify the offset to the end (and start of the following counters/
    // updates/finals arrays).
    DefaultEnd = 8,
    // The following 7 exprs are used by worksharing loops only.
    IsLastIterVariableOffset = 8,
    LowerBoundVariableOffset = 9,
    UpperBoundVariableOffset = 10,
    StrideVariableOffset = 11,
    EnsureUpperBoundOffset = 12,
    NextLowerBoundOffset = 13,
    NextUpperBoundOffset = 14,
    // Offset to the end (and start of the following counters/updates/finals
    // arrays) for worksharing loop directives.
    WorksharingEnd = 15,
  };

  /// \brief Get the counters storage.
  MutableArrayRef<Expr *> getCounters() {
    Expr **Storage = reinterpret_cast<Expr **>(
        &(*(std::next(child_begin(), getArraysOffset(getDirectiveKind())))));
    return MutableArrayRef<Expr *>(Storage, CollapsedNum);
  }

  /// \brief Get the private counters storage.
  MutableArrayRef<Expr *> getPrivateCounters() {
    Expr **Storage = reinterpret_cast<Expr **>(&*std::next(
        child_begin(), getArraysOffset(getDirectiveKind()) + CollapsedNum));
    return MutableArrayRef<Expr *>(Storage, CollapsedNum);
  }

  /// \brief Get the updates storage.
  MutableArrayRef<Expr *> getInits() {
    Expr **Storage = reinterpret_cast<Expr **>(
        &*std::next(child_begin(),
                    getArraysOffset(getDirectiveKind()) + 2 * CollapsedNum));
    return MutableArrayRef<Expr *>(Storage, CollapsedNum);
  }

  /// \brief Get the updates storage.
  MutableArrayRef<Expr *> getUpdates() {
    Expr **Storage = reinterpret_cast<Expr **>(
        &*std::next(child_begin(),
                    getArraysOffset(getDirectiveKind()) + 3 * CollapsedNum));
    return MutableArrayRef<Expr *>(Storage, CollapsedNum);
  }

  /// \brief Get the final counter updates storage.
  MutableArrayRef<Expr *> getFinals() {
    Expr **Storage = reinterpret_cast<Expr **>(
        &*std::next(child_begin(),
                    getArraysOffset(getDirectiveKind()) + 4 * CollapsedNum));
    return MutableArrayRef<Expr *>(Storage, CollapsedNum);
  }

protected:
  /// \brief Build instance of loop directive of class \a Kind.
  ///
  /// \param SC Statement class.
  /// \param Kind Kind of OpenMP directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed loops from 'collapse' clause.
  /// \param NumClauses Number of clauses.
  /// \param NumSpecialChildren Number of additional directive-specific stmts.
  ///
  template <typename T>
  OMPLoopDirective(const T *That, StmtClass SC, OpenMPDirectiveKind Kind,
                   SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned CollapsedNum, unsigned NumClauses,
                   unsigned NumSpecialChildren = 0)
      : OMPExecutableDirective(That, SC, Kind, StartLoc, EndLoc, NumClauses,
                               numLoopChildren(CollapsedNum, Kind) +
                                   NumSpecialChildren),
        CollapsedNum(CollapsedNum) {}

  /// \brief Offset to the start of children expression arrays.
  static unsigned getArraysOffset(OpenMPDirectiveKind Kind) {
    return (isOpenMPWorksharingDirective(Kind) ||
            isOpenMPTaskLoopDirective(Kind) ||
            isOpenMPDistributeDirective(Kind))
               ? WorksharingEnd
               : DefaultEnd;
  }

  /// \brief Children number.
  static unsigned numLoopChildren(unsigned CollapsedNum,
                                  OpenMPDirectiveKind Kind) {
    return getArraysOffset(Kind) + 5 * CollapsedNum; // Counters,
                                                     // PrivateCounters, Inits,
                                                     // Updates and Finals
  }

  void setIterationVariable(Expr *IV) {
    *std::next(child_begin(), IterationVariableOffset) = IV;
  }
  void setLastIteration(Expr *LI) {
    *std::next(child_begin(), LastIterationOffset) = LI;
  }
  void setCalcLastIteration(Expr *CLI) {
    *std::next(child_begin(), CalcLastIterationOffset) = CLI;
  }
  void setPreCond(Expr *PC) {
    *std::next(child_begin(), PreConditionOffset) = PC;
  }
  void setCond(Expr *Cond) {
    *std::next(child_begin(), CondOffset) = Cond;
  }
  void setInit(Expr *Init) { *std::next(child_begin(), InitOffset) = Init; }
  void setInc(Expr *Inc) { *std::next(child_begin(), IncOffset) = Inc; }
  void setIsLastIterVariable(Expr *IL) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), IsLastIterVariableOffset) = IL;
  }
  void setLowerBoundVariable(Expr *LB) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), LowerBoundVariableOffset) = LB;
  }
  void setUpperBoundVariable(Expr *UB) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), UpperBoundVariableOffset) = UB;
  }
  void setStrideVariable(Expr *ST) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), StrideVariableOffset) = ST;
  }
  void setEnsureUpperBound(Expr *EUB) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), EnsureUpperBoundOffset) = EUB;
  }
  void setNextLowerBound(Expr *NLB) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), NextLowerBoundOffset) = NLB;
  }
  void setNextUpperBound(Expr *NUB) {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind()) ||
            isOpenMPDistributeDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    *std::next(child_begin(), NextUpperBoundOffset) = NUB;
  }
  void setCounters(ArrayRef<Expr *> A);
  void setPrivateCounters(ArrayRef<Expr *> A);
  void setInits(ArrayRef<Expr *> A);
  void setUpdates(ArrayRef<Expr *> A);
  void setFinals(ArrayRef<Expr *> A);

public:
  /// \brief The expressions built for the OpenMP loop CodeGen for the
  /// whole collapsed loop nest.
  struct HelperExprs {
    /// \brief Loop iteration variable.
    Expr *IterationVarRef;
    /// \brief Loop last iteration number.
    Expr *LastIteration;
    /// \brief Loop number of iterations.
    Expr *NumIterations;
    /// \brief Calculation of last iteration.
    Expr *CalcLastIteration;
    /// \brief Loop pre-condition.
    Expr *PreCond;
    /// \brief Loop condition.
    Expr *Cond;
    /// \brief Loop iteration variable init.
    Expr *Init;
    /// \brief Loop increment.
    Expr *Inc;
    /// \brief IsLastIteration - local flag variable passed to runtime.
    Expr *IL;
    /// \brief LowerBound - local variable passed to runtime.
    Expr *LB;
    /// \brief UpperBound - local variable passed to runtime.
    Expr *UB;
    /// \brief Stride - local variable passed to runtime.
    Expr *ST;
    /// \brief EnsureUpperBound -- expression LB = min(LB, NumIterations).
    Expr *EUB;
    /// \brief Update of LowerBound for statically sheduled 'omp for' loops.
    Expr *NLB;
    /// \brief Update of UpperBound for statically sheduled 'omp for' loops.
    Expr *NUB;
    /// \brief Counters Loop counters.
    SmallVector<Expr *, 4> Counters;
    /// \brief PrivateCounters Loop counters.
    SmallVector<Expr *, 4> PrivateCounters;
    /// \brief Expressions for loop counters inits for CodeGen.
    SmallVector<Expr *, 4> Inits;
    /// \brief Expressions for loop counters update for CodeGen.
    SmallVector<Expr *, 4> Updates;
    /// \brief Final loop counter values for GodeGen.
    SmallVector<Expr *, 4> Finals;

    /// \brief Check if all the expressions are built (does not check the
    /// worksharing ones).
    bool builtAll() {
      return IterationVarRef != nullptr && LastIteration != nullptr &&
             NumIterations != nullptr && PreCond != nullptr &&
             Cond != nullptr && Init != nullptr && Inc != nullptr;
    }

    /// \brief Initialize all the fields to null.
    /// \param Size Number of elements in the counters/finals/updates arrays.
    void clear(unsigned Size) {
      IterationVarRef = nullptr;
      LastIteration = nullptr;
      CalcLastIteration = nullptr;
      PreCond = nullptr;
      Cond = nullptr;
      Init = nullptr;
      Inc = nullptr;
      IL = nullptr;
      LB = nullptr;
      UB = nullptr;
      ST = nullptr;
      EUB = nullptr;
      NLB = nullptr;
      NUB = nullptr;
      Counters.resize(Size);
      PrivateCounters.resize(Size);
      Inits.resize(Size);
      Updates.resize(Size);
      Finals.resize(Size);
      for (unsigned i = 0; i < Size; ++i) {
        Counters[i] = nullptr;
        PrivateCounters[i] = nullptr;
        Inits[i] = nullptr;
        Updates[i] = nullptr;
        Finals[i] = nullptr;
      }
    }
  };

  /// \brief Get number of collapsed loops.
  unsigned getCollapsedNumber() const { return CollapsedNum; }

  Expr *getIterationVariable() const {
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), IterationVariableOffset)));
  }
  Expr *getLastIteration() const {
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), LastIterationOffset)));
  }
  Expr *getCalcLastIteration() const {
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), CalcLastIterationOffset)));
  }
  Expr *getPreCond() const {
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), PreConditionOffset)));
  }
  Expr *getCond() const {
    return const_cast<Expr *>(
        reinterpret_cast<const Expr *>(*std::next(child_begin(), CondOffset)));
  }
  Expr *getInit() const {
    return const_cast<Expr *>(
        reinterpret_cast<const Expr *>(*std::next(child_begin(), InitOffset)));
  }
  Expr *getInc() const {
    return const_cast<Expr *>(
        reinterpret_cast<const Expr *>(*std::next(child_begin(), IncOffset)));
  }
  Expr *getIsLastIterVariable() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), IsLastIterVariableOffset)));
  }
  Expr *getLowerBoundVariable() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), LowerBoundVariableOffset)));
  }
  Expr *getUpperBoundVariable() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), UpperBoundVariableOffset)));
  }
  Expr *getStrideVariable() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), StrideVariableOffset)));
  }
  Expr *getEnsureUpperBound() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), EnsureUpperBoundOffset)));
  }
  Expr *getNextLowerBound() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), NextLowerBoundOffset)));
  }
  Expr *getNextUpperBound() const {
    assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
            isOpenMPTaskLoopDirective(getDirectiveKind())) &&
           "expected worksharing loop directive");
    return const_cast<Expr *>(reinterpret_cast<const Expr *>(
        *std::next(child_begin(), NextUpperBoundOffset)));
  }
  const Stmt *getBody() const {
    // This relies on the loop form is already checked by Sema.
    Stmt *Body = getAssociatedStmt()->IgnoreContainers(true);
    Body = cast<ForStmt>(Body)->getBody();
    for (unsigned Cnt = 1; Cnt < CollapsedNum; ++Cnt) {
      Body = Body->IgnoreContainers();
      Body = cast<ForStmt>(Body)->getBody();
    }
    return Body;
  }

  ArrayRef<Expr *> counters() { return getCounters(); }

  ArrayRef<Expr *> counters() const {
    return const_cast<OMPLoopDirective *>(this)->getCounters();
  }

  ArrayRef<Expr *> private_counters() { return getPrivateCounters(); }

  ArrayRef<Expr *> private_counters() const {
    return const_cast<OMPLoopDirective *>(this)->getPrivateCounters();
  }

  ArrayRef<Expr *> inits() { return getInits(); }

  ArrayRef<Expr *> inits() const {
    return const_cast<OMPLoopDirective *>(this)->getInits();
  }

  ArrayRef<Expr *> updates() { return getUpdates(); }

  ArrayRef<Expr *> updates() const {
    return const_cast<OMPLoopDirective *>(this)->getUpdates();
  }

  ArrayRef<Expr *> finals() { return getFinals(); }

  ArrayRef<Expr *> finals() const {
    return const_cast<OMPLoopDirective *>(this)->getFinals();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSimdDirectiveClass ||
           T->getStmtClass() == OMPForDirectiveClass ||
           T->getStmtClass() == OMPForSimdDirectiveClass ||
           T->getStmtClass() == OMPParallelForDirectiveClass ||
           T->getStmtClass() == OMPParallelForSimdDirectiveClass ||
           T->getStmtClass() == OMPTaskLoopDirectiveClass ||
           T->getStmtClass() == OMPTaskLoopSimdDirectiveClass ||
           T->getStmtClass() == OMPDistributeDirectiveClass;
  }
};

/// \brief This represents '#pragma omp simd' directive.
///
/// \code
/// #pragma omp simd private(a,b) linear(i,j:s) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp simd' has clauses 'private'
/// with the variables 'a' and 'b', 'linear' with variables 'i', 'j' and
/// linear step 's', 'reduction' with operator '+' and variables 'c' and 'd'.
///
class OMPSimdDirective : public OMPLoopDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPSimdDirectiveClass, OMPD_simd, StartLoc,
                         EndLoc, CollapsedNum, NumClauses) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPSimdDirectiveClass, OMPD_simd,
                         SourceLocation(), SourceLocation(), CollapsedNum,
                         NumClauses) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OMPSimdDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc, unsigned CollapsedNum,
                                  ArrayRef<OMPClause *> Clauses,
                                  Stmt *AssociatedStmt,
                                  const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSimdDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                       unsigned CollapsedNum, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSimdDirectiveClass;
  }
};

/// \brief This represents '#pragma omp for' directive.
///
/// \code
/// #pragma omp for private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp for' has clauses 'private' with the
/// variables 'a' and 'b' and 'reduction' with operator '+' and variables 'c'
/// and 'd'.
///
class OMPForDirective : public OMPLoopDirective {
  friend class ASTStmtReader;

  /// \brief true if current directive has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                  unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPForDirectiveClass, OMPD_for, StartLoc, EndLoc,
                         CollapsedNum, NumClauses),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPForDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPForDirectiveClass, OMPD_for, SourceLocation(),
                         SourceLocation(), CollapsedNum, NumClauses),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  /// \param HasCancel true if current directive has inner cancel directive.
  ///
  static OMPForDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation EndLoc, unsigned CollapsedNum,
                                 ArrayRef<OMPClause *> Clauses,
                                 Stmt *AssociatedStmt, const HelperExprs &Exprs,
                                 bool HasCancel);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPForDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                      unsigned CollapsedNum, EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPForDirectiveClass;
  }
};

/// \brief This represents '#pragma omp for simd' directive.
///
/// \code
/// #pragma omp for simd private(a,b) linear(i,j:s) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp for simd' has clauses 'private'
/// with the variables 'a' and 'b', 'linear' with variables 'i', 'j' and
/// linear step 's', 'reduction' with operator '+' and variables 'c' and 'd'.
///
class OMPForSimdDirective : public OMPLoopDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPForSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                      unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPForSimdDirectiveClass, OMPD_for_simd,
                         StartLoc, EndLoc, CollapsedNum, NumClauses) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPForSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPForSimdDirectiveClass, OMPD_for_simd,
                         SourceLocation(), SourceLocation(), CollapsedNum,
                         NumClauses) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OMPForSimdDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPForSimdDirective *CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses,
                                          unsigned CollapsedNum, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPForSimdDirectiveClass;
  }
};

/// \brief This represents '#pragma omp sections' directive.
///
/// \code
/// #pragma omp sections private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp sections' has clauses 'private' with
/// the variables 'a' and 'b' and 'reduction' with operator '+' and variables
/// 'c' and 'd'.
///
class OMPSectionsDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;

  /// \brief true if current directive has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
                               StartLoc, EndLoc, NumClauses, 1),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSectionsDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param HasCancel true if current directive has inner directive.
  ///
  static OMPSectionsDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSectionsDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSectionsDirectiveClass;
  }
};

/// \brief This represents '#pragma omp section' directive.
///
/// \code
/// #pragma omp section
/// \endcode
///
class OMPSectionDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;

  /// \brief true if current directive has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPSectionDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
                               StartLoc, EndLoc, 0, 1),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPSectionDirective()
      : OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
                               SourceLocation(), SourceLocation(), 0, 1),
        HasCancel(false) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param HasCancel true if current directive has inner directive.
  ///
  static OMPSectionDirective *Create(const ASTContext &C,
                                     SourceLocation StartLoc,
                                     SourceLocation EndLoc,
                                     Stmt *AssociatedStmt, bool HasCancel);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPSectionDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSectionDirectiveClass;
  }
};

/// \brief This represents '#pragma omp single' directive.
///
/// \code
/// #pragma omp single private(a,b) copyprivate(c,d)
/// \endcode
/// In this example directive '#pragma omp single' has clauses 'private' with
/// the variables 'a' and 'b' and 'copyprivate' with variables 'c' and 'd'.
///
class OMPSingleDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPSingleDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSingleDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPSingleDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSingleDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSingleDirectiveClass;
  }
};

/// \brief This represents '#pragma omp master' directive.
///
/// \code
/// #pragma omp master
/// \endcode
///
class OMPMasterDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPMasterDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
                               StartLoc, EndLoc, 0, 1) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPMasterDirective()
      : OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
                               SourceLocation(), SourceLocation(), 0, 1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPMasterDirective *Create(const ASTContext &C,
                                    SourceLocation StartLoc,
                                    SourceLocation EndLoc,
                                    Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPMasterDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPMasterDirectiveClass;
  }
};

/// \brief This represents '#pragma omp critical' directive.
///
/// \code
/// #pragma omp critical
/// \endcode
///
class OMPCriticalDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Name of the directive.
  DeclarationNameInfo DirName;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPCriticalDirective(const DeclarationNameInfo &Name, SourceLocation StartLoc,
                       SourceLocation EndLoc, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
                               StartLoc, EndLoc, NumClauses, 1),
        DirName(Name) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPCriticalDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        DirName() {}

  /// \brief Set name of the directive.
  ///
  /// \param Name Name of the directive.
  ///
  void setDirectiveName(const DeclarationNameInfo &Name) { DirName = Name; }

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPCriticalDirective *
  Create(const ASTContext &C, const DeclarationNameInfo &Name,
         SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPCriticalDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  /// \brief Return name of the directive.
  ///
  DeclarationNameInfo getDirectiveName() const { return DirName; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPCriticalDirectiveClass;
  }
};

/// \brief This represents '#pragma omp parallel for' directive.
///
/// \code
/// #pragma omp parallel for private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp parallel for' has clauses 'private'
/// with the variables 'a' and 'b' and 'reduction' with operator '+' and
/// variables 'c' and 'd'.
///
class OMPParallelForDirective : public OMPLoopDirective {
  friend class ASTStmtReader;

  /// \brief true if current region has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPParallelForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                          unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPParallelForDirectiveClass, OMPD_parallel_for,
                         StartLoc, EndLoc, CollapsedNum, NumClauses),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelForDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPParallelForDirectiveClass, OMPD_parallel_for,
                         SourceLocation(), SourceLocation(), CollapsedNum,
                         NumClauses),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  /// \param HasCancel true if current directive has inner cancel directive.
  ///
  static OMPParallelForDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelForDirective *CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses,
                                              unsigned CollapsedNum,
                                              EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelForDirectiveClass;
  }
};

/// \brief This represents '#pragma omp parallel for simd' directive.
///
/// \code
/// #pragma omp parallel for simd private(a,b) linear(i,j:s) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp parallel for simd' has clauses
/// 'private' with the variables 'a' and 'b', 'linear' with variables 'i', 'j'
/// and linear step 's', 'reduction' with operator '+' and variables 'c' and
/// 'd'.
///
class OMPParallelForSimdDirective : public OMPLoopDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPParallelForSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                              unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPParallelForSimdDirectiveClass,
                         OMPD_parallel_for_simd, StartLoc, EndLoc, CollapsedNum,
                         NumClauses) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelForSimdDirective(unsigned CollapsedNum,
                                       unsigned NumClauses)
      : OMPLoopDirective(this, OMPParallelForSimdDirectiveClass,
                         OMPD_parallel_for_simd, SourceLocation(),
                         SourceLocation(), CollapsedNum, NumClauses) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OMPParallelForSimdDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelForSimdDirective *CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  unsigned CollapsedNum,
                                                  EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelForSimdDirectiveClass;
  }
};

/// \brief This represents '#pragma omp parallel sections' directive.
///
/// \code
/// #pragma omp parallel sections private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp parallel sections' has clauses
/// 'private' with the variables 'a' and 'b' and 'reduction' with operator '+'
/// and variables 'c' and 'd'.
///
class OMPParallelSectionsDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;

  /// \brief true if current directive has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPParallelSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                               unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
                               OMPD_parallel_sections, StartLoc, EndLoc,
                               NumClauses, 1),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelSectionsDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
                               OMPD_parallel_sections, SourceLocation(),
                               SourceLocation(), NumClauses, 1),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param HasCancel true if current directive has inner cancel directive.
  ///
  static OMPParallelSectionsDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelSectionsDirective *
  CreateEmpty(const ASTContext &C, unsigned NumClauses, EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelSectionsDirectiveClass;
  }
};

/// \brief This represents '#pragma omp task' directive.
///
/// \code
/// #pragma omp task private(a,b) final(d)
/// \endcode
/// In this example directive '#pragma omp task' has clauses 'private' with the
/// variables 'a' and 'b' and 'final' with condition 'd'.
///
class OMPTaskDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief true if this directive has inner cancel directive.
  bool HasCancel;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPTaskDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task, StartLoc,
                               EndLoc, NumClauses, 1),
        HasCancel(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTaskDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        HasCancel(false) {}

  /// \brief Set cancel state.
  void setHasCancel(bool Has) { HasCancel = Has; }

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param HasCancel true, if current directive has inner cancel directive.
  ///
  static OMPTaskDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OMPClause *> Clauses,
                                  Stmt *AssociatedStmt, bool HasCancel);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTaskDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                       EmptyShell);

  /// \brief Return true if current directive has inner cancel directive.
  bool hasCancel() const { return HasCancel; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskyield' directive.
///
/// \code
/// #pragma omp taskyield
/// \endcode
///
class OMPTaskyieldDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPTaskyieldDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPTaskyieldDirective()
      : OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPTaskyieldDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPTaskyieldDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskyieldDirectiveClass;
  }
};

/// \brief This represents '#pragma omp barrier' directive.
///
/// \code
/// #pragma omp barrier
/// \endcode
///
class OMPBarrierDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPBarrierDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPBarrierDirective()
      : OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPBarrierDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPBarrierDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPBarrierDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskwait' directive.
///
/// \code
/// #pragma omp taskwait
/// \endcode
///
class OMPTaskwaitDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPTaskwaitDirective()
      : OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPTaskwaitDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPTaskwaitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskwaitDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskgroup' directive.
///
/// \code
/// #pragma omp taskgroup
/// \endcode
///
class OMPTaskgroupDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPTaskgroupDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPTaskgroupDirectiveClass, OMPD_taskgroup,
                               StartLoc, EndLoc, 0, 1) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPTaskgroupDirective()
      : OMPExecutableDirective(this, OMPTaskgroupDirectiveClass, OMPD_taskgroup,
                               SourceLocation(), SourceLocation(), 0, 1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTaskgroupDirective *Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation EndLoc,
                                       Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPTaskgroupDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskgroupDirectiveClass;
  }
};

/// \brief This represents '#pragma omp flush' directive.
///
/// \code
/// #pragma omp flush(a,b)
/// \endcode
/// In this example directive '#pragma omp flush' has 2 arguments- variables 'a'
/// and 'b'.
/// 'omp flush' directive does not have clauses but have an optional list of
/// variables to flush. This list of variables is stored within some fake clause
/// FlushClause.
class OMPFlushDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPFlushDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                    unsigned NumClauses)
      : OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
                               StartLoc, EndLoc, NumClauses, 0) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPFlushDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
                               SourceLocation(), SourceLocation(), NumClauses,
                               0) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses (only single OMPFlushClause clause is
  /// allowed).
  ///
  static OMPFlushDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation EndLoc,
                                   ArrayRef<OMPClause *> Clauses);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPFlushDirective *CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPFlushDirectiveClass;
  }
};

/// \brief This represents '#pragma omp ordered' directive.
///
/// \code
/// #pragma omp ordered
/// \endcode
///
class OMPOrderedDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPOrderedDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                      unsigned NumClauses)
      : OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPOrderedDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPOrderedDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPOrderedDirective *CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPOrderedDirectiveClass;
  }
};

/// \brief This represents '#pragma omp atomic' directive.
///
/// \code
/// #pragma omp atomic capture
/// \endcode
/// In this example directive '#pragma omp atomic' has clause 'capture'.
///
class OMPAtomicDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Used for 'atomic update' or 'atomic capture' constructs. They may
  /// have atomic expressions of forms
  /// \code
  /// x = x binop expr;
  /// x = expr binop x;
  /// \endcode
  /// This field is true for the first form of the expression and false for the
  /// second. Required for correct codegen of non-associative operations (like
  /// << or >>).
  bool IsXLHSInRHSPart;
  /// \brief Used for 'atomic update' or 'atomic capture' constructs. They may
  /// have atomic expressions of forms
  /// \code
  /// v = x; <update x>;
  /// <update x>; v = x;
  /// \endcode
  /// This field is true for the first(postfix) form of the expression and false
  /// otherwise.
  bool IsPostfixUpdate;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPAtomicDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
                               StartLoc, EndLoc, NumClauses, 5),
        IsXLHSInRHSPart(false), IsPostfixUpdate(false) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPAtomicDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
                               SourceLocation(), SourceLocation(), NumClauses,
                               5),
        IsXLHSInRHSPart(false), IsPostfixUpdate(false) {}

  /// \brief Set 'x' part of the associated expression/statement.
  void setX(Expr *X) { *std::next(child_begin()) = X; }
  /// \brief Set helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  void setUpdateExpr(Expr *UE) { *std::next(child_begin(), 2) = UE; }
  /// \brief Set 'v' part of the associated expression/statement.
  void setV(Expr *V) { *std::next(child_begin(), 3) = V; }
  /// \brief Set 'expr' part of the associated expression/statement.
  void setExpr(Expr *E) { *std::next(child_begin(), 4) = E; }

public:
  /// \brief Creates directive with a list of \a Clauses and 'x', 'v' and 'expr'
  /// parts of the atomic construct (see Section 2.12.6, atomic Construct, for
  /// detailed description of 'x', 'v' and 'expr').
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param X 'x' part of the associated expression/statement.
  /// \param V 'v' part of the associated expression/statement.
  /// \param E 'expr' part of the associated expression/statement.
  /// \param UE Helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  /// \param IsXLHSInRHSPart true if \a UE has the first form and false if the
  /// second.
  /// \param IsPostfixUpdate true if original value of 'x' must be stored in
  /// 'v', not an updated one.
  static OMPAtomicDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *X, Expr *V,
         Expr *E, Expr *UE, bool IsXLHSInRHSPart, bool IsPostfixUpdate);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPAtomicDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  /// \brief Get 'x' part of the associated expression/statement.
  Expr *getX() { return cast_or_null<Expr>(*std::next(child_begin())); }
  const Expr *getX() const {
    return cast_or_null<Expr>(*std::next(child_begin()));
  }
  /// \brief Get helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *getUpdateExpr() {
    return cast_or_null<Expr>(*std::next(child_begin(), 2));
  }
  const Expr *getUpdateExpr() const {
    return cast_or_null<Expr>(*std::next(child_begin(), 2));
  }
  /// \brief Return true if helper update expression has form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' and false if it has form
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  bool isXLHSInRHSPart() const { return IsXLHSInRHSPart; }
  /// \brief Return true if 'v' expression must be updated to original value of
  /// 'x', false if 'v' must be updated to the new value of 'x'.
  bool isPostfixUpdate() const { return IsPostfixUpdate; }
  /// \brief Get 'v' part of the associated expression/statement.
  Expr *getV() { return cast_or_null<Expr>(*std::next(child_begin(), 3)); }
  const Expr *getV() const {
    return cast_or_null<Expr>(*std::next(child_begin(), 3));
  }
  /// \brief Get 'expr' part of the associated expression/statement.
  Expr *getExpr() { return cast_or_null<Expr>(*std::next(child_begin(), 4)); }
  const Expr *getExpr() const {
    return cast_or_null<Expr>(*std::next(child_begin(), 4));
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPAtomicDirectiveClass;
  }
};

/// \brief This represents '#pragma omp target' directive.
///
/// \code
/// #pragma omp target if(a)
/// \endcode
/// In this example directive '#pragma omp target' has clause 'if' with
/// condition 'a'.
///
class OMPTargetDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPTargetDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetDirectiveClass, OMPD_target,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTargetDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetDirectiveClass, OMPD_target,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTargetDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTargetDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTargetDirectiveClass;
  }
};

/// \brief This represents '#pragma omp target data' directive.
///
/// \code
/// #pragma omp target data device(0) if(a) map(b[:])
/// \endcode
/// In this example directive '#pragma omp target data' has clauses 'device'
/// with the value '0', 'if' with condition 'a' and 'map' with array
/// section 'b[:]'.
///
class OMPTargetDataDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param NumClauses The number of clauses.
  ///
  OMPTargetDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                         unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetDataDirectiveClass, 
                               OMPD_target_data, StartLoc, EndLoc, NumClauses,
                               1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTargetDataDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetDataDirectiveClass, 
                               OMPD_target_data, SourceLocation(),
                               SourceLocation(), NumClauses, 1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTargetDataDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param N The number of clauses.
  ///
  static OMPTargetDataDirective *CreateEmpty(const ASTContext &C, unsigned N,
                                             EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTargetDataDirectiveClass;
  }
};

/// \brief This represents '#pragma omp target enter data' directive.
///
/// \code
/// #pragma omp target enter data device(0) if(a) map(b[:])
/// \endcode
/// In this example directive '#pragma omp target enter data' has clauses
/// 'device' with the value '0', 'if' with condition 'a' and 'map' with array
/// section 'b[:]'.
///
class OMPTargetEnterDataDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param NumClauses The number of clauses.
  ///
  OMPTargetEnterDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                              unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetEnterDataDirectiveClass,
                               OMPD_target_enter_data, StartLoc, EndLoc,
                               NumClauses, /*NumChildren=*/0) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTargetEnterDataDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetEnterDataDirectiveClass,
                               OMPD_target_enter_data, SourceLocation(),
                               SourceLocation(), NumClauses,
                               /*NumChildren=*/0) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OMPTargetEnterDataDirective *Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param N The number of clauses.
  ///
  static OMPTargetEnterDataDirective *CreateEmpty(const ASTContext &C,
                                                  unsigned N, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTargetEnterDataDirectiveClass;
  }
};

/// \brief This represents '#pragma omp target exit data' directive.
///
/// \code
/// #pragma omp target exit data device(0) if(a) map(b[:])
/// \endcode
/// In this example directive '#pragma omp target exit data' has clauses
/// 'device' with the value '0', 'if' with condition 'a' and 'map' with array
/// section 'b[:]'.
///
class OMPTargetExitDataDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param NumClauses The number of clauses.
  ///
  OMPTargetExitDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                             unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetExitDataDirectiveClass,
                               OMPD_target_exit_data, StartLoc, EndLoc,
                               NumClauses, /*NumChildren=*/0) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTargetExitDataDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetExitDataDirectiveClass,
                               OMPD_target_exit_data, SourceLocation(),
                               SourceLocation(), NumClauses,
                               /*NumChildren=*/0) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OMPTargetExitDataDirective *Create(const ASTContext &C,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            ArrayRef<OMPClause *> Clauses);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param N The number of clauses.
  ///
  static OMPTargetExitDataDirective *CreateEmpty(const ASTContext &C,
                                                 unsigned N, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTargetExitDataDirectiveClass;
  }
};

/// \brief This represents '#pragma omp target parallel' directive.
///
/// \code
/// #pragma omp target parallel if(a)
/// \endcode
/// In this example directive '#pragma omp target parallel' has clause 'if' with
/// condition 'a'.
///
class OMPTargetParallelDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPTargetParallelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                             unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetParallelDirectiveClass,
                               OMPD_target_parallel, StartLoc, EndLoc,
                               NumClauses, /*NumChildren=*/1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTargetParallelDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTargetParallelDirectiveClass,
                               OMPD_target_parallel, SourceLocation(),
                               SourceLocation(), NumClauses,
                               /*NumChildren=*/1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTargetParallelDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTargetParallelDirective *
  CreateEmpty(const ASTContext &C, unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTargetParallelDirectiveClass;
  }
};

/// \brief This represents '#pragma omp teams' directive.
///
/// \code
/// #pragma omp teams if(a)
/// \endcode
/// In this example directive '#pragma omp teams' has clause 'if' with
/// condition 'a'.
///
class OMPTeamsDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPTeamsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                    unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTeamsDirectiveClass, OMPD_teams,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTeamsDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTeamsDirectiveClass, OMPD_teams,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTeamsDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation EndLoc,
                                   ArrayRef<OMPClause *> Clauses,
                                   Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTeamsDirective *CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTeamsDirectiveClass;
  }
};

/// \brief This represents '#pragma omp cancellation point' directive.
///
/// \code
/// #pragma omp cancellation point for
/// \endcode
///
/// In this example a cancellation point is created for innermost 'for' region.
class OMPCancellationPointDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  OpenMPDirectiveKind CancelRegion;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPCancellationPointDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPCancellationPointDirectiveClass,
                               OMPD_cancellation_point, StartLoc, EndLoc, 0, 0),
        CancelRegion(OMPD_unknown) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPCancellationPointDirective()
      : OMPExecutableDirective(this, OMPCancellationPointDirectiveClass,
                               OMPD_cancellation_point, SourceLocation(),
                               SourceLocation(), 0, 0),
        CancelRegion(OMPD_unknown) {}

  /// \brief Set cancel region for current cancellation point.
  /// \param CR Cancellation region.
  void setCancelRegion(OpenMPDirectiveKind CR) { CancelRegion = CR; }

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPCancellationPointDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         OpenMPDirectiveKind CancelRegion);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPCancellationPointDirective *CreateEmpty(const ASTContext &C,
                                                    EmptyShell);

  /// \brief Get cancellation region for the current cancellation point.
  OpenMPDirectiveKind getCancelRegion() const { return CancelRegion; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPCancellationPointDirectiveClass;
  }
};

/// \brief This represents '#pragma omp cancel' directive.
///
/// \code
/// #pragma omp cancel for
/// \endcode
///
/// In this example a cancel is created for innermost 'for' region.
class OMPCancelDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  OpenMPDirectiveKind CancelRegion;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPCancelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPCancelDirectiveClass, OMPD_cancel,
                               StartLoc, EndLoc, NumClauses, 0),
        CancelRegion(OMPD_unknown) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  explicit OMPCancelDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPCancelDirectiveClass, OMPD_cancel,
                               SourceLocation(), SourceLocation(), NumClauses,
                               0),
        CancelRegion(OMPD_unknown) {}

  /// \brief Set cancel region for current cancellation point.
  /// \param CR Cancellation region.
  void setCancelRegion(OpenMPDirectiveKind CR) { CancelRegion = CR; }

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OMPCancelDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, OpenMPDirectiveKind CancelRegion);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPCancelDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  /// \brief Get cancellation region for the current cancellation point.
  OpenMPDirectiveKind getCancelRegion() const { return CancelRegion; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPCancelDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskloop' directive.
///
/// \code
/// #pragma omp taskloop private(a,b) grainsize(val) num_tasks(num)
/// \endcode
/// In this example directive '#pragma omp taskloop' has clauses 'private'
/// with the variables 'a' and 'b', 'grainsize' with expression 'val' and
/// 'num_tasks' with expression 'num'.
///
class OMPTaskLoopDirective : public OMPLoopDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPTaskLoopDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPTaskLoopDirectiveClass, OMPD_taskloop,
                         StartLoc, EndLoc, CollapsedNum, NumClauses) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTaskLoopDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPTaskLoopDirectiveClass, OMPD_taskloop,
                         SourceLocation(), SourceLocation(), CollapsedNum,
                         NumClauses) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OMPTaskLoopDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTaskLoopDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses,
                                           unsigned CollapsedNum, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskLoopDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskloop simd' directive.
///
/// \code
/// #pragma omp taskloop simd private(a,b) grainsize(val) num_tasks(num)
/// \endcode
/// In this example directive '#pragma omp taskloop simd' has clauses 'private'
/// with the variables 'a' and 'b', 'grainsize' with expression 'val' and
/// 'num_tasks' with expression 'num'.
///
class OMPTaskLoopSimdDirective : public OMPLoopDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPTaskLoopSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                           unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPTaskLoopSimdDirectiveClass,
                         OMPD_taskloop_simd, StartLoc, EndLoc, CollapsedNum,
                         NumClauses) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTaskLoopSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPTaskLoopSimdDirectiveClass,
                         OMPD_taskloop_simd, SourceLocation(), SourceLocation(),
                         CollapsedNum, NumClauses) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OMPTaskLoopSimdDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTaskLoopSimdDirective *CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses,
                                               unsigned CollapsedNum,
                                               EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskLoopSimdDirectiveClass;
  }
};

/// \brief This represents '#pragma omp distribute' directive.
///
/// \code
/// #pragma omp distribute private(a,b)
/// \endcode
/// In this example directive '#pragma omp distribute' has clauses 'private'
/// with the variables 'a' and 'b'
///
class OMPDistributeDirective : public OMPLoopDirective {
  friend class ASTStmtReader;

  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPDistributeDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                         unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPDistributeDirectiveClass, OMPD_distribute,
                         StartLoc, EndLoc, CollapsedNum, NumClauses)
        {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPDistributeDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPLoopDirective(this, OMPDistributeDirectiveClass, OMPD_distribute,
                         SourceLocation(), SourceLocation(), CollapsedNum,
                         NumClauses)
        {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
    ///
  static OMPDistributeDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt, const HelperExprs &Exprs);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPDistributeDirective *CreateEmpty(const ASTContext &C,
                                             unsigned NumClauses,
                                             unsigned CollapsedNum, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPDistributeDirectiveClass;
  }
};

} // end namespace clang

#endif
