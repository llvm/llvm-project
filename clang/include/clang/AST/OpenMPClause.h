//===- OpenMPClause.h - Classes for OpenMP clauses --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file defines OpenMP AST classes for clauses.
/// There are clauses for executable directives, clauses for declarative
/// directives and clauses which can be used in both kinds of directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENMPCLAUSE_H
#define LLVM_CLANG_AST_OPENMPCLAUSE_H

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for clauses.
//===----------------------------------------------------------------------===//

/// \brief This is a basic class for representing single OpenMP clause.
///
class OMPClause {
  /// \brief Starting location of the clause (the clause keyword).
  SourceLocation StartLoc;
  /// \brief Ending location of the clause.
  SourceLocation EndLoc;
  /// \brief Kind of the clause.
  OpenMPClauseKind Kind;

protected:
  OMPClause(OpenMPClauseKind K, SourceLocation StartLoc, SourceLocation EndLoc)
      : StartLoc(StartLoc), EndLoc(EndLoc), Kind(K) {}

public:
  /// \brief Returns the starting location of the clause.
  SourceLocation getLocStart() const { return StartLoc; }
  /// \brief Returns the ending location of the clause.
  SourceLocation getLocEnd() const { return EndLoc; }

  /// \brief Sets the starting location of the clause.
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// \brief Sets the ending location of the clause.
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// \brief Returns kind of OpenMP clause (private, shared, reduction, etc.).
  OpenMPClauseKind getClauseKind() const { return Kind; }

  bool isImplicit() const { return StartLoc.isInvalid(); }

  typedef StmtIterator child_iterator;
  typedef ConstStmtIterator const_child_iterator;
  typedef llvm::iterator_range<child_iterator> child_range;
  typedef llvm::iterator_range<const_child_iterator> const_child_range;

  child_range children();
  const_child_range children() const {
    auto Children = const_cast<OMPClause *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }
  static bool classof(const OMPClause *) { return true; }
};

/// Class that handles pre-initialization statement for some clauses, like
/// 'shedule', 'firstprivate' etc.
class OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// Pre-initialization statement for the clause.
  Stmt *PreInit;
  /// Region that captures the associated stmt.
  OpenMPDirectiveKind CaptureRegion;

protected:
  /// Set pre-initialization statement for the clause.
  void setPreInitStmt(Stmt *S, OpenMPDirectiveKind ThisRegion = OMPD_unknown) {
    PreInit = S;
    CaptureRegion = ThisRegion;
  }
  OMPClauseWithPreInit(const OMPClause *This)
      : PreInit(nullptr), CaptureRegion(OMPD_unknown) {
    assert(get(This) && "get is not tuned for pre-init.");
  }

public:
  /// Get pre-initialization statement for the clause.
  const Stmt *getPreInitStmt() const { return PreInit; }
  /// Get pre-initialization statement for the clause.
  Stmt *getPreInitStmt() { return PreInit; }
  /// Get capture region for the stmt in the clause.
  OpenMPDirectiveKind getCaptureRegion() { return CaptureRegion; }
  static OMPClauseWithPreInit *get(OMPClause *C);
  static const OMPClauseWithPreInit *get(const OMPClause *C);
};

/// Class that handles post-update expression for some clauses, like
/// 'lastprivate', 'reduction' etc.
class OMPClauseWithPostUpdate : public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// Post-update expression for the clause.
  Expr *PostUpdate;
protected:
  /// Set pre-initialization statement for the clause.
  void setPostUpdateExpr(Expr *S) { PostUpdate = S; }
  OMPClauseWithPostUpdate(const OMPClause *This)
      : OMPClauseWithPreInit(This), PostUpdate(nullptr) {
    assert(get(This) && "get is not tuned for post-update.");
  }

public:
  /// Get post-update expression for the clause.
  const Expr *getPostUpdateExpr() const { return PostUpdate; }
  /// Get post-update expression for the clause.
  Expr *getPostUpdateExpr() { return PostUpdate; }
  static OMPClauseWithPostUpdate *get(OMPClause *C);
  static const OMPClauseWithPostUpdate *get(const OMPClause *C);
};

/// \brief This represents clauses with the list of variables like 'private',
/// 'firstprivate', 'copyin', 'shared', or 'reduction' clauses in the
/// '#pragma omp ...' directives.
template <class T> class OMPVarListClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of variables in the list.
  unsigned NumVars;

protected:
  /// \brief Fetches list of variables associated with this clause.
  MutableArrayRef<Expr *> getVarRefs() {
    return MutableArrayRef<Expr *>(
        static_cast<T *>(this)->template getTrailingObjects<Expr *>(), NumVars);
  }

  /// \brief Sets the list of variables for this clause.
  void setVarRefs(ArrayRef<Expr *> VL) {
    assert(VL.size() == NumVars &&
           "Number of variables is not the same as the preallocated buffer");
    std::copy(VL.begin(), VL.end(),
              static_cast<T *>(this)->template getTrailingObjects<Expr *>());
  }

  /// \brief Build a clause with \a N variables
  ///
  /// \param K Kind of the clause.
  /// \param StartLoc Starting location of the clause (the clause keyword).
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPVarListClause(OpenMPClauseKind K, SourceLocation StartLoc,
                   SourceLocation LParenLoc, SourceLocation EndLoc, unsigned N)
      : OMPClause(K, StartLoc, EndLoc), LParenLoc(LParenLoc), NumVars(N) {}

public:
  typedef MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
  typedef llvm::iterator_range<varlist_iterator> varlist_range;
  typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }

  varlist_iterator varlist_begin() { return getVarRefs().begin(); }
  varlist_iterator varlist_end() { return getVarRefs().end(); }
  varlist_const_iterator varlist_begin() const { return getVarRefs().begin(); }
  varlist_const_iterator varlist_end() const { return getVarRefs().end(); }

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Fetches list of all variables in the clause.
  ArrayRef<const Expr *> getVarRefs() const {
    return llvm::makeArrayRef(
        static_cast<const T *>(this)->template getTrailingObjects<Expr *>(),
        NumVars);
  }
};

/// \brief This represents 'if' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp parallel if(parallel:a > 5)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'if' clause with
/// condition 'a > 5' and directive name modifier 'parallel'.
///
class OMPIfClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Condition of the 'if' clause.
  Stmt *Condition;
  /// \brief Location of ':' (if any).
  SourceLocation ColonLoc;
  /// \brief Directive name modifier for the clause.
  OpenMPDirectiveKind NameModifier;
  /// \brief Name modifier location.
  SourceLocation NameModifierLoc;

  /// \brief Set condition.
  ///
  void setCondition(Expr *Cond) { Condition = Cond; }
  /// \brief Set directive name modifier for the clause.
  ///
  void setNameModifier(OpenMPDirectiveKind NM) { NameModifier = NM; }
  /// \brief Set location of directive name modifier for the clause.
  ///
  void setNameModifierLoc(SourceLocation Loc) { NameModifierLoc = Loc; }
  /// \brief Set location of ':'.
  ///
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
  /// \brief Build 'if' clause with condition \a Cond.
  ///
  /// \param NameModifier [OpenMP 4.1] Directive name modifier of clause.
  /// \param Cond Condition of the clause.
  /// \param HelperCond Helper condition for the clause.
  /// \param CaptureRegion Innermost OpenMP region where expressions in this
  /// clause must be captured.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param NameModifierLoc Location of directive name modifier.
  /// \param ColonLoc [OpenMP 4.1] Location of ':'.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPIfClause(OpenMPDirectiveKind NameModifier, Expr *Cond, Stmt *HelperCond,
              OpenMPDirectiveKind CaptureRegion, SourceLocation StartLoc,
              SourceLocation LParenLoc, SourceLocation NameModifierLoc,
              SourceLocation ColonLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_if, StartLoc, EndLoc), OMPClauseWithPreInit(this),
        LParenLoc(LParenLoc), Condition(Cond), ColonLoc(ColonLoc),
        NameModifier(NameModifier), NameModifierLoc(NameModifierLoc) {
    setPreInitStmt(HelperCond, CaptureRegion);
  }

  /// \brief Build an empty clause.
  ///
  OMPIfClause()
      : OMPClause(OMPC_if, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), LParenLoc(), Condition(nullptr), ColonLoc(),
        NameModifier(OMPD_unknown), NameModifierLoc() {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return the location of ':'.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// \brief Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }
  /// \brief Return directive name modifier associated with the clause.
  OpenMPDirectiveKind getNameModifier() const { return NameModifier; }

  /// \brief Return the location of directive name modifier.
  SourceLocation getNameModifierLoc() const { return NameModifierLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_if;
  }

  child_range children() { return child_range(&Condition, &Condition + 1); }
};

/// \brief This represents 'final' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp task final(a > 5)
/// \endcode
/// In this example directive '#pragma omp task' has simple 'final'
/// clause with condition 'a > 5'.
///
class OMPFinalClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Condition of the 'if' clause.
  Stmt *Condition;

  /// \brief Set condition.
  ///
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// \brief Build 'final' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPFinalClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc)
      : OMPClause(OMPC_final, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// \brief Build an empty clause.
  ///
  OMPFinalClause()
      : OMPClause(OMPC_final, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Condition(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_final;
  }

  child_range children() { return child_range(&Condition, &Condition + 1); }
};

/// \brief This represents 'num_threads' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp parallel num_threads(6)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'num_threads'
/// clause with number of threads '6'.
///
class OMPNumThreadsClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Condition of the 'num_threads' clause.
  Stmt *NumThreads;

  /// \brief Set condition.
  ///
  void setNumThreads(Expr *NThreads) { NumThreads = NThreads; }

public:
  /// \brief Build 'num_threads' clause with condition \a NumThreads.
  ///
  /// \param NumThreads Number of threads for the construct.
  /// \param HelperNumThreads Helper Number of threads for the construct.
  /// \param CaptureRegion Innermost OpenMP region where expressions in this
  /// clause must be captured.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNumThreadsClause(Expr *NumThreads, Stmt *HelperNumThreads,
                      OpenMPDirectiveKind CaptureRegion,
                      SourceLocation StartLoc, SourceLocation LParenLoc,
                      SourceLocation EndLoc)
      : OMPClause(OMPC_num_threads, StartLoc, EndLoc),
        OMPClauseWithPreInit(this), LParenLoc(LParenLoc),
        NumThreads(NumThreads) {
    setPreInitStmt(HelperNumThreads, CaptureRegion);
  }

  /// \brief Build an empty clause.
  ///
  OMPNumThreadsClause()
      : OMPClause(OMPC_num_threads, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), LParenLoc(SourceLocation()),
        NumThreads(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns number of threads.
  Expr *getNumThreads() const { return cast_or_null<Expr>(NumThreads); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_num_threads;
  }

  child_range children() { return child_range(&NumThreads, &NumThreads + 1); }
};

/// \brief This represents 'safelen' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp simd safelen(4)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'safelen'
/// with single expression '4'.
/// If the safelen clause is used then no two iterations executed
/// concurrently with SIMD instructions can have a greater distance
/// in the logical iteration space than its value. The parameter of
/// the safelen clause must be a constant positive integer expression.
///
class OMPSafelenClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Safe iteration space distance.
  Stmt *Safelen;

  /// \brief Set safelen.
  void setSafelen(Expr *Len) { Safelen = Len; }

public:
  /// \brief Build 'safelen' clause.
  ///
  /// \param Len Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPSafelenClause(Expr *Len, SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_safelen, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Safelen(Len) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPSafelenClause()
      : OMPClause(OMPC_safelen, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Safelen(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return safe iteration space distance.
  Expr *getSafelen() const { return cast_or_null<Expr>(Safelen); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_safelen;
  }

  child_range children() { return child_range(&Safelen, &Safelen + 1); }
};

/// \brief This represents 'simdlen' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp simd simdlen(4)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'simdlen'
/// with single expression '4'.
/// If the 'simdlen' clause is used then it specifies the preferred number of
/// iterations to be executed concurrently. The parameter of the 'simdlen'
/// clause must be a constant positive integer expression.
///
class OMPSimdlenClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Safe iteration space distance.
  Stmt *Simdlen;

  /// \brief Set simdlen.
  void setSimdlen(Expr *Len) { Simdlen = Len; }

public:
  /// \brief Build 'simdlen' clause.
  ///
  /// \param Len Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPSimdlenClause(Expr *Len, SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_simdlen, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Simdlen(Len) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPSimdlenClause()
      : OMPClause(OMPC_simdlen, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Simdlen(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return safe iteration space distance.
  Expr *getSimdlen() const { return cast_or_null<Expr>(Simdlen); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_simdlen;
  }

  child_range children() { return child_range(&Simdlen, &Simdlen + 1); }
};

/// \brief This represents 'collapse' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp simd collapse(3)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'collapse'
/// with single expression '3'.
/// The parameter must be a constant positive integer expression, it specifies
/// the number of nested loops that should be collapsed into a single iteration
/// space.
///
class OMPCollapseClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of for-loops.
  Stmt *NumForLoops;

  /// \brief Set the number of associated for-loops.
  void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
  /// \brief Build 'collapse' clause.
  ///
  /// \param Num Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPCollapseClause(Expr *Num, SourceLocation StartLoc,
                    SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_collapse, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumForLoops(Num) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPCollapseClause()
      : OMPClause(OMPC_collapse, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumForLoops(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return the number of associated for-loops.
  Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_collapse;
  }

  child_range children() { return child_range(&NumForLoops, &NumForLoops + 1); }
};

/// \brief This represents 'default' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp parallel default(shared)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'default'
/// clause with kind 'shared'.
///
class OMPDefaultClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'default' clause.
  OpenMPDefaultClauseKind Kind;
  /// \brief Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// \brief Set kind of the clauses.
  ///
  /// \param K Argument of clause.
  ///
  void setDefaultKind(OpenMPDefaultClauseKind K) { Kind = K; }

  /// \brief Set argument location.
  ///
  /// \param KLoc Argument location.
  ///
  void setDefaultKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// \brief Build 'default' clause with argument \a A ('none' or 'shared').
  ///
  /// \param A Argument of the clause ('none' or 'shared').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPDefaultClause(OpenMPDefaultClauseKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_default, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// \brief Build an empty clause.
  ///
  OMPDefaultClause()
      : OMPClause(OMPC_default, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Kind(OMPC_DEFAULT_unknown),
        KindKwLoc(SourceLocation()) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns kind of the clause.
  OpenMPDefaultClauseKind getDefaultKind() const { return Kind; }

  /// \brief Returns location of clause kind.
  SourceLocation getDefaultKindKwLoc() const { return KindKwLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_default;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'proc_bind' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp parallel proc_bind(master)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'proc_bind'
/// clause with kind 'master'.
///
class OMPProcBindClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'proc_bind' clause.
  OpenMPProcBindClauseKind Kind;
  /// \brief Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// \brief Set kind of the clause.
  ///
  /// \param K Kind of clause.
  ///
  void setProcBindKind(OpenMPProcBindClauseKind K) { Kind = K; }

  /// \brief Set clause kind location.
  ///
  /// \param KLoc Kind location.
  ///
  void setProcBindKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// \brief Build 'proc_bind' clause with argument \a A ('master', 'close' or
  ///        'spread').
  ///
  /// \param A Argument of the clause ('master', 'close' or 'spread').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPProcBindClause(OpenMPProcBindClauseKind A, SourceLocation ALoc,
                    SourceLocation StartLoc, SourceLocation LParenLoc,
                    SourceLocation EndLoc)
      : OMPClause(OMPC_proc_bind, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// \brief Build an empty clause.
  ///
  OMPProcBindClause()
      : OMPClause(OMPC_proc_bind, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Kind(OMPC_PROC_BIND_unknown),
        KindKwLoc(SourceLocation()) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns kind of the clause.
  OpenMPProcBindClauseKind getProcBindKind() const { return Kind; }

  /// \brief Returns location of clause kind.
  SourceLocation getProcBindKindKwLoc() const { return KindKwLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_proc_bind;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'schedule' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp for schedule(static, 3)
/// \endcode
/// In this example directive '#pragma omp for' has 'schedule' clause with
/// arguments 'static' and '3'.
///
class OMPScheduleClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'schedule' clause.
  OpenMPScheduleClauseKind Kind;
  /// \brief Modifiers for 'schedule' clause.
  enum {FIRST, SECOND, NUM_MODIFIERS};
  OpenMPScheduleClauseModifier Modifiers[NUM_MODIFIERS];
  /// \brief Locations of modifiers.
  SourceLocation ModifiersLoc[NUM_MODIFIERS];
  /// \brief Start location of the schedule ind in source code.
  SourceLocation KindLoc;
  /// \brief Location of ',' (if any).
  SourceLocation CommaLoc;
  /// \brief Chunk size.
  Expr *ChunkSize;

  /// \brief Set schedule kind.
  ///
  /// \param K Schedule kind.
  ///
  void setScheduleKind(OpenMPScheduleClauseKind K) { Kind = K; }
  /// \brief Set the first schedule modifier.
  ///
  /// \param M Schedule modifier.
  ///
  void setFirstScheduleModifier(OpenMPScheduleClauseModifier M) {
    Modifiers[FIRST] = M;
  }
  /// \brief Set the second schedule modifier.
  ///
  /// \param M Schedule modifier.
  ///
  void setSecondScheduleModifier(OpenMPScheduleClauseModifier M) {
    Modifiers[SECOND] = M;
  }
  /// \brief Set location of the first schedule modifier.
  ///
  void setFirstScheduleModifierLoc(SourceLocation Loc) {
    ModifiersLoc[FIRST] = Loc;
  }
  /// \brief Set location of the second schedule modifier.
  ///
  void setSecondScheduleModifierLoc(SourceLocation Loc) {
    ModifiersLoc[SECOND] = Loc;
  }
  /// \brief Set schedule modifier location.
  ///
  /// \param M Schedule modifier location.
  ///
  void setScheduleModifer(OpenMPScheduleClauseModifier M) {
    if (Modifiers[FIRST] == OMPC_SCHEDULE_MODIFIER_unknown)
      Modifiers[FIRST] = M;
    else {
      assert(Modifiers[SECOND] == OMPC_SCHEDULE_MODIFIER_unknown);
      Modifiers[SECOND] = M;
    }
  }
  /// \brief Sets the location of '('.
  ///
  /// \param Loc Location of '('.
  ///
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Set schedule kind start location.
  ///
  /// \param KLoc Schedule kind location.
  ///
  void setScheduleKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }
  /// \brief Set location of ','.
  ///
  /// \param Loc Location of ','.
  ///
  void setCommaLoc(SourceLocation Loc) { CommaLoc = Loc; }
  /// \brief Set chunk size.
  ///
  /// \param E Chunk size.
  ///
  void setChunkSize(Expr *E) { ChunkSize = E; }

public:
  /// \brief Build 'schedule' clause with schedule kind \a Kind and chunk size
  /// expression \a ChunkSize.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param KLoc Starting location of the argument.
  /// \param CommaLoc Location of ','.
  /// \param EndLoc Ending location of the clause.
  /// \param Kind Schedule kind.
  /// \param ChunkSize Chunk size.
  /// \param HelperChunkSize Helper chunk size for combined directives.
  /// \param M1 The first modifier applied to 'schedule' clause.
  /// \param M1Loc Location of the first modifier
  /// \param M2 The second modifier applied to 'schedule' clause.
  /// \param M2Loc Location of the second modifier
  ///
  OMPScheduleClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                    SourceLocation KLoc, SourceLocation CommaLoc,
                    SourceLocation EndLoc, OpenMPScheduleClauseKind Kind,
                    Expr *ChunkSize, Stmt *HelperChunkSize,
                    OpenMPScheduleClauseModifier M1, SourceLocation M1Loc,
                    OpenMPScheduleClauseModifier M2, SourceLocation M2Loc)
      : OMPClause(OMPC_schedule, StartLoc, EndLoc), OMPClauseWithPreInit(this),
        LParenLoc(LParenLoc), Kind(Kind), KindLoc(KLoc), CommaLoc(CommaLoc),
        ChunkSize(ChunkSize) {
    setPreInitStmt(HelperChunkSize);
    Modifiers[FIRST] = M1;
    Modifiers[SECOND] = M2;
    ModifiersLoc[FIRST] = M1Loc;
    ModifiersLoc[SECOND] = M2Loc;
  }

  /// \brief Build an empty clause.
  ///
  explicit OMPScheduleClause()
      : OMPClause(OMPC_schedule, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), Kind(OMPC_SCHEDULE_unknown),
        ChunkSize(nullptr) {
    Modifiers[FIRST] = OMPC_SCHEDULE_MODIFIER_unknown;
    Modifiers[SECOND] = OMPC_SCHEDULE_MODIFIER_unknown;
  }

  /// \brief Get kind of the clause.
  ///
  OpenMPScheduleClauseKind getScheduleKind() const { return Kind; }
  /// \brief Get the first modifier of the clause.
  ///
  OpenMPScheduleClauseModifier getFirstScheduleModifier() const {
    return Modifiers[FIRST];
  }
  /// \brief Get the second modifier of the clause.
  ///
  OpenMPScheduleClauseModifier getSecondScheduleModifier() const {
    return Modifiers[SECOND];
  }
  /// \brief Get location of '('.
  ///
  SourceLocation getLParenLoc() { return LParenLoc; }
  /// \brief Get kind location.
  ///
  SourceLocation getScheduleKindLoc() { return KindLoc; }
  /// \brief Get the first modifier location.
  ///
  SourceLocation getFirstScheduleModifierLoc() const {
    return ModifiersLoc[FIRST];
  }
  /// \brief Get the second modifier location.
  ///
  SourceLocation getSecondScheduleModifierLoc() const {
    return ModifiersLoc[SECOND];
  }
  /// \brief Get location of ','.
  ///
  SourceLocation getCommaLoc() { return CommaLoc; }
  /// \brief Get chunk size.
  ///
  Expr *getChunkSize() { return ChunkSize; }
  /// \brief Get chunk size.
  ///
  const Expr *getChunkSize() const { return ChunkSize; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_schedule;
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(&ChunkSize),
                       reinterpret_cast<Stmt **>(&ChunkSize) + 1);
  }
};

/// \brief This represents 'ordered' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp for ordered (2)
/// \endcode
/// In this example directive '#pragma omp for' has 'ordered' clause with
/// parameter 2.
///
class OMPOrderedClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of for-loops.
  Stmt *NumForLoops;

  /// \brief Set the number of associated for-loops.
  void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
  /// \brief Build 'ordered' clause.
  ///
  /// \param Num Expression, possibly associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPOrderedClause(Expr *Num, SourceLocation StartLoc,
                    SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_ordered, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumForLoops(Num) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPOrderedClause()
      : OMPClause(OMPC_ordered, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumForLoops(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return the number of associated for-loops.
  Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_ordered;
  }

  child_range children() { return child_range(&NumForLoops, &NumForLoops + 1); }
};

/// \brief This represents 'nowait' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp for nowait
/// \endcode
/// In this example directive '#pragma omp for' has 'nowait' clause.
///
class OMPNowaitClause : public OMPClause {
public:
  /// \brief Build 'nowait' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNowaitClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_nowait, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPNowaitClause()
      : OMPClause(OMPC_nowait, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_nowait;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'untied' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp task untied
/// \endcode
/// In this example directive '#pragma omp task' has 'untied' clause.
///
class OMPUntiedClause : public OMPClause {
public:
  /// \brief Build 'untied' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPUntiedClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_untied, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPUntiedClause()
      : OMPClause(OMPC_untied, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_untied;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'mergeable' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp task mergeable
/// \endcode
/// In this example directive '#pragma omp task' has 'mergeable' clause.
///
class OMPMergeableClause : public OMPClause {
public:
  /// \brief Build 'mergeable' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPMergeableClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_mergeable, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPMergeableClause()
      : OMPClause(OMPC_mergeable, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_mergeable;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'read' clause in the '#pragma omp atomic' directive.
///
/// \code
/// #pragma omp atomic read
/// \endcode
/// In this example directive '#pragma omp atomic' has 'read' clause.
///
class OMPReadClause : public OMPClause {
public:
  /// \brief Build 'read' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPReadClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_read, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPReadClause() : OMPClause(OMPC_read, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_read;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'write' clause in the '#pragma omp atomic' directive.
///
/// \code
/// #pragma omp atomic write
/// \endcode
/// In this example directive '#pragma omp atomic' has 'write' clause.
///
class OMPWriteClause : public OMPClause {
public:
  /// \brief Build 'write' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPWriteClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_write, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPWriteClause()
      : OMPClause(OMPC_write, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_write;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'update' clause in the '#pragma omp atomic'
/// directive.
///
/// \code
/// #pragma omp atomic update
/// \endcode
/// In this example directive '#pragma omp atomic' has 'update' clause.
///
class OMPUpdateClause : public OMPClause {
public:
  /// \brief Build 'update' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPUpdateClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_update, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPUpdateClause()
      : OMPClause(OMPC_update, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_update;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'capture' clause in the '#pragma omp atomic'
/// directive.
///
/// \code
/// #pragma omp atomic capture
/// \endcode
/// In this example directive '#pragma omp atomic' has 'capture' clause.
///
class OMPCaptureClause : public OMPClause {
public:
  /// \brief Build 'capture' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPCaptureClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_capture, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPCaptureClause()
      : OMPClause(OMPC_capture, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_capture;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'seq_cst' clause in the '#pragma omp atomic'
/// directive.
///
/// \code
/// #pragma omp atomic seq_cst
/// \endcode
/// In this example directive '#pragma omp atomic' has 'seq_cst' clause.
///
class OMPSeqCstClause : public OMPClause {
public:
  /// \brief Build 'seq_cst' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPSeqCstClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_seq_cst, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPSeqCstClause()
      : OMPClause(OMPC_seq_cst, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_seq_cst;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents clause 'private' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel private(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'private'
/// with the variables 'a' and 'b'.
///
class OMPPrivateClause final
    : public OMPVarListClause<OMPPrivateClause>,
      private llvm::TrailingObjects<OMPPrivateClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPPrivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPPrivateClause>(OMPC_private, StartLoc, LParenLoc,
                                           EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPPrivateClause(unsigned N)
      : OMPVarListClause<OMPPrivateClause>(OMPC_private, SourceLocation(),
                                           SourceLocation(), SourceLocation(),
                                           N) {}

  /// \brief Sets the list of references to private copies with initializers for
  /// new private variables.
  /// \param VL List of references.
  void setPrivateCopies(ArrayRef<Expr *> VL);

  /// \brief Gets the list of references to private copies with initializers for
  /// new private variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param PrivateVL List of references to private copies with initializers.
  ///
  static OMPPrivateClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                  ArrayRef<Expr *> PrivateVL);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPPrivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator private_copies_iterator;
  typedef ArrayRef<const Expr *>::iterator private_copies_const_iterator;
  typedef llvm::iterator_range<private_copies_iterator> private_copies_range;
  typedef llvm::iterator_range<private_copies_const_iterator>
      private_copies_const_range;

  private_copies_range private_copies() {
    return private_copies_range(getPrivateCopies().begin(),
                                getPrivateCopies().end());
  }
  private_copies_const_range private_copies() const {
    return private_copies_const_range(getPrivateCopies().begin(),
                                      getPrivateCopies().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_private;
  }
};

/// \brief This represents clause 'firstprivate' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp parallel firstprivate(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'firstprivate'
/// with the variables 'a' and 'b'.
///
class OMPFirstprivateClause final
    : public OMPVarListClause<OMPFirstprivateClause>,
      public OMPClauseWithPreInit,
      private llvm::TrailingObjects<OMPFirstprivateClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;

  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPFirstprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                        SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPFirstprivateClause>(OMPC_firstprivate, StartLoc,
                                                LParenLoc, EndLoc, N),
        OMPClauseWithPreInit(this) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPFirstprivateClause(unsigned N)
      : OMPVarListClause<OMPFirstprivateClause>(
            OMPC_firstprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N),
        OMPClauseWithPreInit(this) {}
  /// \brief Sets the list of references to private copies with initializers for
  /// new private variables.
  /// \param VL List of references.
  void setPrivateCopies(ArrayRef<Expr *> VL);

  /// \brief Gets the list of references to private copies with initializers for
  /// new private variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// \brief Sets the list of references to initializer variables for new
  /// private variables.
  /// \param VL List of references.
  void setInits(ArrayRef<Expr *> VL);

  /// \brief Gets the list of references to initializer variables for new
  /// private variables.
  MutableArrayRef<Expr *> getInits() {
    return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
  }
  ArrayRef<const Expr *> getInits() const {
    return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the original variables.
  /// \param PrivateVL List of references to private copies with initializers.
  /// \param InitVL List of references to auto generated variables used for
  /// initialization of a single array element. Used if firstprivate variable is
  /// of array type.
  /// \param PreInit Statement that must be executed before entering the OpenMP
  /// region with this clause.
  ///
  static OMPFirstprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
         ArrayRef<Expr *> InitVL, Stmt *PreInit);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPFirstprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator private_copies_iterator;
  typedef ArrayRef<const Expr *>::iterator private_copies_const_iterator;
  typedef llvm::iterator_range<private_copies_iterator> private_copies_range;
  typedef llvm::iterator_range<private_copies_const_iterator>
      private_copies_const_range;

  private_copies_range private_copies() {
    return private_copies_range(getPrivateCopies().begin(),
                                getPrivateCopies().end());
  }
  private_copies_const_range private_copies() const {
    return private_copies_const_range(getPrivateCopies().begin(),
                                      getPrivateCopies().end());
  }

  typedef MutableArrayRef<Expr *>::iterator inits_iterator;
  typedef ArrayRef<const Expr *>::iterator inits_const_iterator;
  typedef llvm::iterator_range<inits_iterator> inits_range;
  typedef llvm::iterator_range<inits_const_iterator> inits_const_range;

  inits_range inits() {
    return inits_range(getInits().begin(), getInits().end());
  }
  inits_const_range inits() const {
    return inits_const_range(getInits().begin(), getInits().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_firstprivate;
  }
};

/// \brief This represents clause 'lastprivate' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp simd lastprivate(a,b)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'lastprivate'
/// with the variables 'a' and 'b'.
class OMPLastprivateClause final
    : public OMPVarListClause<OMPLastprivateClause>,
      public OMPClauseWithPostUpdate,
      private llvm::TrailingObjects<OMPLastprivateClause, Expr *> {
  // There are 4 additional tail-allocated arrays at the end of the class:
  // 1. Contains list of pseudo variables with the default initialization for
  // each non-firstprivate variables. Used in codegen for initialization of
  // lastprivate copies.
  // 2. List of helper expressions for proper generation of assignment operation
  // required for lastprivate clause. This list represents private variables
  // (for arrays, single array element).
  // 3. List of helper expressions for proper generation of assignment operation
  // required for lastprivate clause. This list represents original variables
  // (for arrays, single array element).
  // 4. List of helper expressions that represents assignment operation:
  // \code
  // DstExprs = SrcExprs;
  // \endcode
  // Required for proper codegen of final assignment performed by the
  // lastprivate clause.
  //
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;

  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPLastprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPLastprivateClause>(OMPC_lastprivate, StartLoc,
                                               LParenLoc, EndLoc, N),
        OMPClauseWithPostUpdate(this) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPLastprivateClause(unsigned N)
      : OMPVarListClause<OMPLastprivateClause>(
            OMPC_lastprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N),
        OMPClauseWithPostUpdate(this) {}

  /// \brief Get the list of helper expressions for initialization of private
  /// copies for lastprivate variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent private variables (for arrays, single
  /// array element) in the final assignment statement performed by the
  /// lastprivate clause.
  void setSourceExprs(ArrayRef<Expr *> SrcExprs);

  /// \brief Get the list of helper source expressions.
  MutableArrayRef<Expr *> getSourceExprs() {
    return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
  }
  ArrayRef<const Expr *> getSourceExprs() const {
    return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent original variables (for arrays, single
  /// array element) in the final assignment statement performed by the
  /// lastprivate clause.
  void setDestinationExprs(ArrayRef<Expr *> DstExprs);

  /// \brief Get the list of helper destination expressions.
  MutableArrayRef<Expr *> getDestinationExprs() {
    return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getDestinationExprs() const {
    return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
  }

  /// \brief Set list of helper assignment expressions, required for proper
  /// codegen of the clause. These expressions are assignment expressions that
  /// assign private copy of the variable to original variable.
  void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

  /// \brief Get the list of helper assignment expressions.
  MutableArrayRef<Expr *> getAssignmentOps() {
    return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getAssignmentOps() const {
    return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param SrcExprs List of helper expressions for proper generation of
  /// assignment operation required for lastprivate clause. This list represents
  /// private variables (for arrays, single array element).
  /// \param DstExprs List of helper expressions for proper generation of
  /// assignment operation required for lastprivate clause. This list represents
  /// original variables (for arrays, single array element).
  /// \param AssignmentOps List of helper expressions that represents assignment
  /// operation:
  /// \code
  /// DstExprs = SrcExprs;
  /// \endcode
  /// Required for proper codegen of final assignment performed by the
  /// lastprivate clause.
  /// \param PreInit Statement that must be executed before entering the OpenMP
  /// region with this clause.
  /// \param PostUpdate Expression that must be executed after exit from the
  /// OpenMP region with this clause.
  ///
  static OMPLastprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
         ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps,
         Stmt *PreInit, Expr *PostUpdate);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPLastprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator helper_expr_iterator;
  typedef ArrayRef<const Expr *>::iterator helper_expr_const_iterator;
  typedef llvm::iterator_range<helper_expr_iterator> helper_expr_range;
  typedef llvm::iterator_range<helper_expr_const_iterator>
      helper_expr_const_range;

  /// \brief Set list of helper expressions, required for generation of private
  /// copies of original lastprivate variables.
  void setPrivateCopies(ArrayRef<Expr *> PrivateCopies);

  helper_expr_const_range private_copies() const {
    return helper_expr_const_range(getPrivateCopies().begin(),
                                   getPrivateCopies().end());
  }
  helper_expr_range private_copies() {
    return helper_expr_range(getPrivateCopies().begin(),
                             getPrivateCopies().end());
  }
  helper_expr_const_range source_exprs() const {
    return helper_expr_const_range(getSourceExprs().begin(),
                                   getSourceExprs().end());
  }
  helper_expr_range source_exprs() {
    return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
  }
  helper_expr_const_range destination_exprs() const {
    return helper_expr_const_range(getDestinationExprs().begin(),
                                   getDestinationExprs().end());
  }
  helper_expr_range destination_exprs() {
    return helper_expr_range(getDestinationExprs().begin(),
                             getDestinationExprs().end());
  }
  helper_expr_const_range assignment_ops() const {
    return helper_expr_const_range(getAssignmentOps().begin(),
                                   getAssignmentOps().end());
  }
  helper_expr_range assignment_ops() {
    return helper_expr_range(getAssignmentOps().begin(),
                             getAssignmentOps().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_lastprivate;
  }
};

/// \brief This represents clause 'shared' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel shared(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'shared'
/// with the variables 'a' and 'b'.
///
class OMPSharedClause final
    : public OMPVarListClause<OMPSharedClause>,
      private llvm::TrailingObjects<OMPSharedClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPSharedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPSharedClause>(OMPC_shared, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPSharedClause(unsigned N)
      : OMPVarListClause<OMPSharedClause>(OMPC_shared, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPSharedClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPSharedClause *CreateEmpty(const ASTContext &C, unsigned N);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_shared;
  }
};

/// \brief This represents clause 'reduction' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp parallel reduction(+:a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'reduction'
/// with operator '+' and the variables 'a' and 'b'.
///
class OMPReductionClause final
    : public OMPVarListClause<OMPReductionClause>,
      public OMPClauseWithPostUpdate,
      private llvm::TrailingObjects<OMPReductionClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Location of ':'.
  SourceLocation ColonLoc;
  /// \brief Nested name specifier for C++.
  NestedNameSpecifierLoc QualifierLoc;
  /// \brief Name of custom operator.
  DeclarationNameInfo NameInfo;

  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param ColonLoc Location of ':'.
  /// \param N Number of the variables in the clause.
  /// \param QualifierLoc The nested-name qualifier with location information
  /// \param NameInfo The full name info for reduction identifier.
  ///
  OMPReductionClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                     SourceLocation ColonLoc, SourceLocation EndLoc, unsigned N,
                     NestedNameSpecifierLoc QualifierLoc,
                     const DeclarationNameInfo &NameInfo)
      : OMPVarListClause<OMPReductionClause>(OMPC_reduction, StartLoc,
                                             LParenLoc, EndLoc, N),
        OMPClauseWithPostUpdate(this), ColonLoc(ColonLoc),
        QualifierLoc(QualifierLoc), NameInfo(NameInfo) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPReductionClause(unsigned N)
      : OMPVarListClause<OMPReductionClause>(OMPC_reduction, SourceLocation(),
                                             SourceLocation(), SourceLocation(),
                                             N),
        OMPClauseWithPostUpdate(this), ColonLoc(), QualifierLoc(), NameInfo() {}

  /// \brief Sets location of ':' symbol in clause.
  void setColonLoc(SourceLocation CL) { ColonLoc = CL; }
  /// \brief Sets the name info for specified reduction identifier.
  void setNameInfo(DeclarationNameInfo DNI) { NameInfo = DNI; }
  /// \brief Sets the nested name specifier.
  void setQualifierLoc(NestedNameSpecifierLoc NSL) { QualifierLoc = NSL; }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent private copy of the reduction
  /// variable.
  void setPrivates(ArrayRef<Expr *> Privates);

  /// \brief Get the list of helper privates.
  MutableArrayRef<Expr *> getPrivates() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivates() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent LHS expression in the final
  /// reduction expression performed by the reduction clause.
  void setLHSExprs(ArrayRef<Expr *> LHSExprs);

  /// \brief Get the list of helper LHS expressions.
  MutableArrayRef<Expr *> getLHSExprs() {
    return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
  }
  ArrayRef<const Expr *> getLHSExprs() const {
    return llvm::makeArrayRef(getPrivates().end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent RHS expression in the final
  /// reduction expression performed by the reduction clause.
  /// Also, variables in these expressions are used for proper initialization of
  /// reduction copies.
  void setRHSExprs(ArrayRef<Expr *> RHSExprs);

  /// \brief Get the list of helper destination expressions.
  MutableArrayRef<Expr *> getRHSExprs() {
    return MutableArrayRef<Expr *>(getLHSExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getRHSExprs() const {
    return llvm::makeArrayRef(getLHSExprs().end(), varlist_size());
  }

  /// \brief Set list of helper reduction expressions, required for proper
  /// codegen of the clause. These expressions are binary expressions or
  /// operator/custom reduction call that calculates new value from source
  /// helper expressions to destination helper expressions.
  void setReductionOps(ArrayRef<Expr *> ReductionOps);

  /// \brief Get the list of helper reduction expressions.
  MutableArrayRef<Expr *> getReductionOps() {
    return MutableArrayRef<Expr *>(getRHSExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getReductionOps() const {
    return llvm::makeArrayRef(getRHSExprs().end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL The variables in the clause.
  /// \param QualifierLoc The nested-name qualifier with location information
  /// \param NameInfo The full name info for reduction identifier.
  /// \param Privates List of helper expressions for proper generation of
  /// private copies.
  /// \param LHSExprs List of helper expressions for proper generation of
  /// assignment operation required for copyprivate clause. This list represents
  /// LHSs of the reduction expressions.
  /// \param RHSExprs List of helper expressions for proper generation of
  /// assignment operation required for copyprivate clause. This list represents
  /// RHSs of the reduction expressions.
  /// Also, variables in these expressions are used for proper initialization of
  /// reduction copies.
  /// \param ReductionOps List of helper expressions that represents reduction
  /// expressions:
  /// \code
  /// LHSExprs binop RHSExprs;
  /// operator binop(LHSExpr, RHSExpr);
  /// <CutomReduction>(LHSExpr, RHSExpr);
  /// \endcode
  /// Required for proper codegen of final reduction operation performed by the
  /// reduction clause.
  /// \param PreInit Statement that must be executed before entering the OpenMP
  /// region with this clause.
  /// \param PostUpdate Expression that must be executed after exit from the
  /// OpenMP region with this clause.
  ///
  static OMPReductionClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
         NestedNameSpecifierLoc QualifierLoc,
         const DeclarationNameInfo &NameInfo, ArrayRef<Expr *> Privates,
         ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
         ArrayRef<Expr *> ReductionOps, Stmt *PreInit, Expr *PostUpdate);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPReductionClause *CreateEmpty(const ASTContext &C, unsigned N);

  /// \brief Gets location of ':' symbol in clause.
  SourceLocation getColonLoc() const { return ColonLoc; }
  /// \brief Gets the name info for specified reduction identifier.
  const DeclarationNameInfo &getNameInfo() const { return NameInfo; }
  /// \brief Gets the nested name specifier.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  typedef MutableArrayRef<Expr *>::iterator helper_expr_iterator;
  typedef ArrayRef<const Expr *>::iterator helper_expr_const_iterator;
  typedef llvm::iterator_range<helper_expr_iterator> helper_expr_range;
  typedef llvm::iterator_range<helper_expr_const_iterator>
      helper_expr_const_range;

  helper_expr_const_range privates() const {
    return helper_expr_const_range(getPrivates().begin(), getPrivates().end());
  }
  helper_expr_range privates() {
    return helper_expr_range(getPrivates().begin(), getPrivates().end());
  }
  helper_expr_const_range lhs_exprs() const {
    return helper_expr_const_range(getLHSExprs().begin(), getLHSExprs().end());
  }
  helper_expr_range lhs_exprs() {
    return helper_expr_range(getLHSExprs().begin(), getLHSExprs().end());
  }
  helper_expr_const_range rhs_exprs() const {
    return helper_expr_const_range(getRHSExprs().begin(), getRHSExprs().end());
  }
  helper_expr_range rhs_exprs() {
    return helper_expr_range(getRHSExprs().begin(), getRHSExprs().end());
  }
  helper_expr_const_range reduction_ops() const {
    return helper_expr_const_range(getReductionOps().begin(),
                                   getReductionOps().end());
  }
  helper_expr_range reduction_ops() {
    return helper_expr_range(getReductionOps().begin(),
                             getReductionOps().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_reduction;
  }
};

/// \brief This represents clause 'linear' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp simd linear(a,b : 2)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'linear'
/// with variables 'a', 'b' and linear step '2'.
///
class OMPLinearClause final
    : public OMPVarListClause<OMPLinearClause>,
      public OMPClauseWithPostUpdate,
      private llvm::TrailingObjects<OMPLinearClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Modifier of 'linear' clause.
  OpenMPLinearClauseKind Modifier;
  /// \brief Location of linear modifier if any.
  SourceLocation ModifierLoc;
  /// \brief Location of ':'.
  SourceLocation ColonLoc;

  /// \brief Sets the linear step for clause.
  void setStep(Expr *Step) { *(getFinals().end()) = Step; }

  /// \brief Sets the expression to calculate linear step for clause.
  void setCalcStep(Expr *CalcStep) { *(getFinals().end() + 1) = CalcStep; }

  /// \brief Build 'linear' clause with given number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of variables.
  ///
  OMPLinearClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
                  SourceLocation ColonLoc, SourceLocation EndLoc,
                  unsigned NumVars)
      : OMPVarListClause<OMPLinearClause>(OMPC_linear, StartLoc, LParenLoc,
                                          EndLoc, NumVars),
        OMPClauseWithPostUpdate(this), Modifier(Modifier),
        ModifierLoc(ModifierLoc), ColonLoc(ColonLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of variables.
  ///
  explicit OMPLinearClause(unsigned NumVars)
      : OMPVarListClause<OMPLinearClause>(OMPC_linear, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          NumVars),
        OMPClauseWithPostUpdate(this), Modifier(OMPC_LINEAR_val), ModifierLoc(),
        ColonLoc() {}

  /// \brief Gets the list of initial values for linear variables.
  ///
  /// There are NumVars expressions with initial values allocated after the
  /// varlist, they are followed by NumVars update expressions (used to update
  /// the linear variable's value on current iteration) and they are followed by
  /// NumVars final expressions (used to calculate the linear variable's
  /// value after the loop body). After these lists, there are 2 helper
  /// expressions - linear step and a helper to calculate it before the
  /// loop body (used when the linear step is not constant):
  ///
  /// { Vars[] /* in OMPVarListClause */; Privates[]; Inits[]; Updates[];
  /// Finals[]; Step; CalcStep; }
  ///
  MutableArrayRef<Expr *> getPrivates() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivates() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  MutableArrayRef<Expr *> getInits() {
    return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
  }
  ArrayRef<const Expr *> getInits() const {
    return llvm::makeArrayRef(getPrivates().end(), varlist_size());
  }

  /// \brief Sets the list of update expressions for linear variables.
  MutableArrayRef<Expr *> getUpdates() {
    return MutableArrayRef<Expr *>(getInits().end(), varlist_size());
  }
  ArrayRef<const Expr *> getUpdates() const {
    return llvm::makeArrayRef(getInits().end(), varlist_size());
  }

  /// \brief Sets the list of final update expressions for linear variables.
  MutableArrayRef<Expr *> getFinals() {
    return MutableArrayRef<Expr *>(getUpdates().end(), varlist_size());
  }
  ArrayRef<const Expr *> getFinals() const {
    return llvm::makeArrayRef(getUpdates().end(), varlist_size());
  }

  /// \brief Sets the list of the copies of original linear variables.
  /// \param PL List of expressions.
  void setPrivates(ArrayRef<Expr *> PL);

  /// \brief Sets the list of the initial values for linear variables.
  /// \param IL List of expressions.
  void setInits(ArrayRef<Expr *> IL);

public:
  /// \brief Creates clause with a list of variables \a VL and a linear step
  /// \a Step.
  ///
  /// \param C AST Context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Modifier Modifier of 'linear' clause.
  /// \param ModifierLoc Modifier location.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param PL List of private copies of original variables.
  /// \param IL List of initial values for the variables.
  /// \param Step Linear step.
  /// \param CalcStep Calculation of the linear step.
  /// \param PreInit Statement that must be executed before entering the OpenMP
  /// region with this clause.
  /// \param PostUpdate Expression that must be executed after exit from the
  /// OpenMP region with this clause.
  static OMPLinearClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
         SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
         ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep,
         Stmt *PreInit, Expr *PostUpdate);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of variables.
  ///
  static OMPLinearClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

  /// \brief Set modifier.
  void setModifier(OpenMPLinearClauseKind Kind) { Modifier = Kind; }
  /// \brief Return modifier.
  OpenMPLinearClauseKind getModifier() const { return Modifier; }

  /// \brief Set modifier location.
  void setModifierLoc(SourceLocation Loc) { ModifierLoc = Loc; }
  /// \brief Return modifier location.
  SourceLocation getModifierLoc() const { return ModifierLoc; }

  /// \brief Sets the location of ':'.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }
  /// \brief Returns the location of ':'.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// \brief Returns linear step.
  Expr *getStep() { return *(getFinals().end()); }
  /// \brief Returns linear step.
  const Expr *getStep() const { return *(getFinals().end()); }
  /// \brief Returns expression to calculate linear step.
  Expr *getCalcStep() { return *(getFinals().end() + 1); }
  /// \brief Returns expression to calculate linear step.
  const Expr *getCalcStep() const { return *(getFinals().end() + 1); }

  /// \brief Sets the list of update expressions for linear variables.
  /// \param UL List of expressions.
  void setUpdates(ArrayRef<Expr *> UL);

  /// \brief Sets the list of final update expressions for linear variables.
  /// \param FL List of expressions.
  void setFinals(ArrayRef<Expr *> FL);

  typedef MutableArrayRef<Expr *>::iterator privates_iterator;
  typedef ArrayRef<const Expr *>::iterator privates_const_iterator;
  typedef llvm::iterator_range<privates_iterator> privates_range;
  typedef llvm::iterator_range<privates_const_iterator> privates_const_range;

  privates_range privates() {
    return privates_range(getPrivates().begin(), getPrivates().end());
  }
  privates_const_range privates() const {
    return privates_const_range(getPrivates().begin(), getPrivates().end());
  }

  typedef MutableArrayRef<Expr *>::iterator inits_iterator;
  typedef ArrayRef<const Expr *>::iterator inits_const_iterator;
  typedef llvm::iterator_range<inits_iterator> inits_range;
  typedef llvm::iterator_range<inits_const_iterator> inits_const_range;

  inits_range inits() {
    return inits_range(getInits().begin(), getInits().end());
  }
  inits_const_range inits() const {
    return inits_const_range(getInits().begin(), getInits().end());
  }

  typedef MutableArrayRef<Expr *>::iterator updates_iterator;
  typedef ArrayRef<const Expr *>::iterator updates_const_iterator;
  typedef llvm::iterator_range<updates_iterator> updates_range;
  typedef llvm::iterator_range<updates_const_iterator> updates_const_range;

  updates_range updates() {
    return updates_range(getUpdates().begin(), getUpdates().end());
  }
  updates_const_range updates() const {
    return updates_const_range(getUpdates().begin(), getUpdates().end());
  }

  typedef MutableArrayRef<Expr *>::iterator finals_iterator;
  typedef ArrayRef<const Expr *>::iterator finals_const_iterator;
  typedef llvm::iterator_range<finals_iterator> finals_range;
  typedef llvm::iterator_range<finals_const_iterator> finals_const_range;

  finals_range finals() {
    return finals_range(getFinals().begin(), getFinals().end());
  }
  finals_const_range finals() const {
    return finals_const_range(getFinals().begin(), getFinals().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_linear;
  }
};

/// \brief This represents clause 'aligned' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp simd aligned(a,b : 8)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'aligned'
/// with variables 'a', 'b' and alignment '8'.
///
class OMPAlignedClause final
    : public OMPVarListClause<OMPAlignedClause>,
      private llvm::TrailingObjects<OMPAlignedClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Location of ':'.
  SourceLocation ColonLoc;

  /// \brief Sets the alignment for clause.
  void setAlignment(Expr *A) { *varlist_end() = A; }

  /// \brief Build 'aligned' clause with given number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of variables.
  ///
  OMPAlignedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation ColonLoc, SourceLocation EndLoc,
                   unsigned NumVars)
      : OMPVarListClause<OMPAlignedClause>(OMPC_aligned, StartLoc, LParenLoc,
                                           EndLoc, NumVars),
        ColonLoc(ColonLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of variables.
  ///
  explicit OMPAlignedClause(unsigned NumVars)
      : OMPVarListClause<OMPAlignedClause>(OMPC_aligned, SourceLocation(),
                                           SourceLocation(), SourceLocation(),
                                           NumVars),
        ColonLoc(SourceLocation()) {}

public:
  /// \brief Creates clause with a list of variables \a VL and alignment \a A.
  ///
  /// \param C AST Context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param A Alignment.
  static OMPAlignedClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation ColonLoc,
                                  SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                  Expr *A);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of variables.
  ///
  static OMPAlignedClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

  /// \brief Sets the location of ':'.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }
  /// \brief Returns the location of ':'.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// \brief Returns alignment.
  Expr *getAlignment() { return *varlist_end(); }
  /// \brief Returns alignment.
  const Expr *getAlignment() const { return *varlist_end(); }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_aligned;
  }
};

/// \brief This represents clause 'copyin' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel copyin(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'copyin'
/// with the variables 'a' and 'b'.
///
class OMPCopyinClause final
    : public OMPVarListClause<OMPCopyinClause>,
      private llvm::TrailingObjects<OMPCopyinClause, Expr *> {
  // Class has 3 additional tail allocated arrays:
  // 1. List of helper expressions for proper generation of assignment operation
  // required for copyin clause. This list represents sources.
  // 2. List of helper expressions for proper generation of assignment operation
  // required for copyin clause. This list represents destinations.
  // 3. List of helper expressions that represents assignment operation:
  // \code
  // DstExprs = SrcExprs;
  // \endcode
  // Required for proper codegen of propagation of master's thread values of
  // threadprivate variables to local instances of that variables in other
  // implicit threads.

  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPCopyinClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPCopyinClause>(OMPC_copyin, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPCopyinClause(unsigned N)
      : OMPVarListClause<OMPCopyinClause>(OMPC_copyin, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent source expression in the final
  /// assignment statement performed by the copyin clause.
  void setSourceExprs(ArrayRef<Expr *> SrcExprs);

  /// \brief Get the list of helper source expressions.
  MutableArrayRef<Expr *> getSourceExprs() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getSourceExprs() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent destination expression in the final
  /// assignment statement performed by the copyin clause.
  void setDestinationExprs(ArrayRef<Expr *> DstExprs);

  /// \brief Get the list of helper destination expressions.
  MutableArrayRef<Expr *> getDestinationExprs() {
    return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getDestinationExprs() const {
    return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
  }

  /// \brief Set list of helper assignment expressions, required for proper
  /// codegen of the clause. These expressions are assignment expressions that
  /// assign source helper expressions to destination helper expressions
  /// correspondingly.
  void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

  /// \brief Get the list of helper assignment expressions.
  MutableArrayRef<Expr *> getAssignmentOps() {
    return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getAssignmentOps() const {
    return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param SrcExprs List of helper expressions for proper generation of
  /// assignment operation required for copyin clause. This list represents
  /// sources.
  /// \param DstExprs List of helper expressions for proper generation of
  /// assignment operation required for copyin clause. This list represents
  /// destinations.
  /// \param AssignmentOps List of helper expressions that represents assignment
  /// operation:
  /// \code
  /// DstExprs = SrcExprs;
  /// \endcode
  /// Required for proper codegen of propagation of master's thread values of
  /// threadprivate variables to local instances of that variables in other
  /// implicit threads.
  ///
  static OMPCopyinClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
         ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPCopyinClause *CreateEmpty(const ASTContext &C, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator helper_expr_iterator;
  typedef ArrayRef<const Expr *>::iterator helper_expr_const_iterator;
  typedef llvm::iterator_range<helper_expr_iterator> helper_expr_range;
  typedef llvm::iterator_range<helper_expr_const_iterator>
      helper_expr_const_range;

  helper_expr_const_range source_exprs() const {
    return helper_expr_const_range(getSourceExprs().begin(),
                                   getSourceExprs().end());
  }
  helper_expr_range source_exprs() {
    return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
  }
  helper_expr_const_range destination_exprs() const {
    return helper_expr_const_range(getDestinationExprs().begin(),
                                   getDestinationExprs().end());
  }
  helper_expr_range destination_exprs() {
    return helper_expr_range(getDestinationExprs().begin(),
                             getDestinationExprs().end());
  }
  helper_expr_const_range assignment_ops() const {
    return helper_expr_const_range(getAssignmentOps().begin(),
                                   getAssignmentOps().end());
  }
  helper_expr_range assignment_ops() {
    return helper_expr_range(getAssignmentOps().begin(),
                             getAssignmentOps().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_copyin;
  }
};

/// \brief This represents clause 'copyprivate' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp single copyprivate(a,b)
/// \endcode
/// In this example directive '#pragma omp single' has clause 'copyprivate'
/// with the variables 'a' and 'b'.
///
class OMPCopyprivateClause final
    : public OMPVarListClause<OMPCopyprivateClause>,
      private llvm::TrailingObjects<OMPCopyprivateClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPCopyprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPCopyprivateClause>(OMPC_copyprivate, StartLoc,
                                               LParenLoc, EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPCopyprivateClause(unsigned N)
      : OMPVarListClause<OMPCopyprivateClause>(
            OMPC_copyprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N) {}

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent source expression in the final
  /// assignment statement performed by the copyprivate clause.
  void setSourceExprs(ArrayRef<Expr *> SrcExprs);

  /// \brief Get the list of helper source expressions.
  MutableArrayRef<Expr *> getSourceExprs() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getSourceExprs() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// \brief Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent destination expression in the final
  /// assignment statement performed by the copyprivate clause.
  void setDestinationExprs(ArrayRef<Expr *> DstExprs);

  /// \brief Get the list of helper destination expressions.
  MutableArrayRef<Expr *> getDestinationExprs() {
    return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getDestinationExprs() const {
    return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
  }

  /// \brief Set list of helper assignment expressions, required for proper
  /// codegen of the clause. These expressions are assignment expressions that
  /// assign source helper expressions to destination helper expressions
  /// correspondingly.
  void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

  /// \brief Get the list of helper assignment expressions.
  MutableArrayRef<Expr *> getAssignmentOps() {
    return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getAssignmentOps() const {
    return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
  }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param SrcExprs List of helper expressions for proper generation of
  /// assignment operation required for copyprivate clause. This list represents
  /// sources.
  /// \param DstExprs List of helper expressions for proper generation of
  /// assignment operation required for copyprivate clause. This list represents
  /// destinations.
  /// \param AssignmentOps List of helper expressions that represents assignment
  /// operation:
  /// \code
  /// DstExprs = SrcExprs;
  /// \endcode
  /// Required for proper codegen of final assignment performed by the
  /// copyprivate clause.
  ///
  static OMPCopyprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
         ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPCopyprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator helper_expr_iterator;
  typedef ArrayRef<const Expr *>::iterator helper_expr_const_iterator;
  typedef llvm::iterator_range<helper_expr_iterator> helper_expr_range;
  typedef llvm::iterator_range<helper_expr_const_iterator>
      helper_expr_const_range;

  helper_expr_const_range source_exprs() const {
    return helper_expr_const_range(getSourceExprs().begin(),
                                   getSourceExprs().end());
  }
  helper_expr_range source_exprs() {
    return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
  }
  helper_expr_const_range destination_exprs() const {
    return helper_expr_const_range(getDestinationExprs().begin(),
                                   getDestinationExprs().end());
  }
  helper_expr_range destination_exprs() {
    return helper_expr_range(getDestinationExprs().begin(),
                             getDestinationExprs().end());
  }
  helper_expr_const_range assignment_ops() const {
    return helper_expr_const_range(getAssignmentOps().begin(),
                                   getAssignmentOps().end());
  }
  helper_expr_range assignment_ops() {
    return helper_expr_range(getAssignmentOps().begin(),
                             getAssignmentOps().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_copyprivate;
  }
};

/// \brief This represents implicit clause 'flush' for the '#pragma omp flush'
/// directive.
/// This clause does not exist by itself, it can be only as a part of 'omp
/// flush' directive. This clause is introduced to keep the original structure
/// of \a OMPExecutableDirective class and its derivatives and to use the
/// existing infrastructure of clauses with the list of variables.
///
/// \code
/// #pragma omp flush(a,b)
/// \endcode
/// In this example directive '#pragma omp flush' has implicit clause 'flush'
/// with the variables 'a' and 'b'.
///
class OMPFlushClause final
    : public OMPVarListClause<OMPFlushClause>,
      private llvm::TrailingObjects<OMPFlushClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPFlushClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPFlushClause>(OMPC_flush, StartLoc, LParenLoc,
                                         EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPFlushClause(unsigned N)
      : OMPVarListClause<OMPFlushClause>(OMPC_flush, SourceLocation(),
                                         SourceLocation(), SourceLocation(),
                                         N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPFlushClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                SourceLocation LParenLoc, SourceLocation EndLoc,
                                ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPFlushClause *CreateEmpty(const ASTContext &C, unsigned N);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_flush;
  }
};

/// \brief This represents implicit clause 'depend' for the '#pragma omp task'
/// directive.
///
/// \code
/// #pragma omp task depend(in:a,b)
/// \endcode
/// In this example directive '#pragma omp task' with clause 'depend' with the
/// variables 'a' and 'b' with dependency 'in'.
///
class OMPDependClause final
    : public OMPVarListClause<OMPDependClause>,
      private llvm::TrailingObjects<OMPDependClause, Expr *> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend class OMPClauseReader;
  /// \brief Dependency type (one of in, out, inout).
  OpenMPDependClauseKind DepKind;
  /// \brief Dependency type location.
  SourceLocation DepLoc;
  /// \brief Colon location.
  SourceLocation ColonLoc;
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPDependClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPDependClause>(OMPC_depend, StartLoc, LParenLoc,
                                          EndLoc, N),
        DepKind(OMPC_DEPEND_unknown) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPDependClause(unsigned N)
      : OMPVarListClause<OMPDependClause>(OMPC_depend, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N),
        DepKind(OMPC_DEPEND_unknown) {}
  /// \brief Set dependency kind.
  void setDependencyKind(OpenMPDependClauseKind K) { DepKind = K; }

  /// \brief Set dependency kind and its location.
  void setDependencyLoc(SourceLocation Loc) { DepLoc = Loc; }

  /// \brief Set colon location.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param DepKind Dependency type.
  /// \param DepLoc Location of the dependency type.
  /// \param ColonLoc Colon location.
  /// \param VL List of references to the variables.
  static OMPDependClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, OpenMPDependClauseKind DepKind,
         SourceLocation DepLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPDependClause *CreateEmpty(const ASTContext &C, unsigned N);

  /// \brief Get dependency type.
  OpenMPDependClauseKind getDependencyKind() const { return DepKind; }
  /// \brief Get dependency type location.
  SourceLocation getDependencyLoc() const { return DepLoc; }
  /// \brief Get colon location.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// Set the loop counter value for the depend clauses with 'sink|source' kind
  /// of dependency. Required for codegen.
  void setCounterValue(Expr *V);
  /// Get the loop counter value.
  Expr *getCounterValue();
  /// Get the loop counter value.
  const Expr *getCounterValue() const;

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_depend;
  }
};

/// \brief This represents 'device' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp target device(a)
/// \endcode
/// In this example directive '#pragma omp target' has clause 'device'
/// with single expression 'a'.
///
class OMPDeviceClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Device number.
  Stmt *Device;
  /// \brief Set the device number.
  ///
  /// \param E Device number.
  ///
  void setDevice(Expr *E) { Device = E; }

public:
  /// \brief Build 'device' clause.
  ///
  /// \param E Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPDeviceClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc, 
                  SourceLocation EndLoc)
      : OMPClause(OMPC_device, StartLoc, EndLoc), LParenLoc(LParenLoc), 
        Device(E) {}

  /// \brief Build an empty clause.
  ///
  OMPDeviceClause()
      : OMPClause(OMPC_device, SourceLocation(), SourceLocation()), 
        LParenLoc(SourceLocation()), Device(nullptr) {}
  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }
  /// \brief Return device number.
  Expr *getDevice() { return cast<Expr>(Device); }
  /// \brief Return device number.
  Expr *getDevice() const { return cast<Expr>(Device); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_device;
  }

  child_range children() { return child_range(&Device, &Device + 1); }
};

/// \brief This represents 'threads' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp ordered threads
/// \endcode
/// In this example directive '#pragma omp ordered' has simple 'threads' clause.
///
class OMPThreadsClause : public OMPClause {
public:
  /// \brief Build 'threads' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPThreadsClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_threads, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPThreadsClause()
      : OMPClause(OMPC_threads, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_threads;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'simd' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp ordered simd
/// \endcode
/// In this example directive '#pragma omp ordered' has simple 'simd' clause.
///
class OMPSIMDClause : public OMPClause {
public:
  /// \brief Build 'simd' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPSIMDClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_simd, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPSIMDClause() : OMPClause(OMPC_simd, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_simd;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief Struct that defines common infrastructure to handle mappable
/// expressions used in OpenMP clauses.
class OMPClauseMappableExprCommon {
public:
  // \brief Class that represents a component of a mappable expression. E.g.
  // for an expression S.a, the first component is a declaration reference
  // expression associated with 'S' and the second is a member expression
  // associated with the field declaration 'a'. If the expression is an array
  // subscript it may not have any associated declaration. In that case the
  // associated declaration is set to nullptr.
  class MappableComponent {
    // \brief Expression associated with the component.
    Expr *AssociatedExpression = nullptr;
    // \brief Declaration associated with the declaration. If the component does
    // not have a declaration (e.g. array subscripts or section), this is set to
    // nullptr.
    ValueDecl *AssociatedDeclaration = nullptr;

  public:
    explicit MappableComponent() {}
    explicit MappableComponent(Expr *AssociatedExpression,
                               ValueDecl *AssociatedDeclaration)
        : AssociatedExpression(AssociatedExpression),
          AssociatedDeclaration(
              AssociatedDeclaration
                  ? cast<ValueDecl>(AssociatedDeclaration->getCanonicalDecl())
                  : nullptr) {}

    Expr *getAssociatedExpression() const { return AssociatedExpression; }
    ValueDecl *getAssociatedDeclaration() const {
      return AssociatedDeclaration;
    }
  };

  // \brief List of components of an expression. This first one is the whole
  // expression and the last one is the base expression.
  typedef SmallVector<MappableComponent, 8> MappableExprComponentList;
  typedef ArrayRef<MappableComponent> MappableExprComponentListRef;

  // \brief List of all component lists associated to the same base declaration.
  // E.g. if both 'S.a' and 'S.b' are a mappable expressions, each will have
  // their component list but the same base declaration 'S'.
  typedef SmallVector<MappableExprComponentList, 8> MappableExprComponentLists;
  typedef ArrayRef<MappableExprComponentList> MappableExprComponentListsRef;

protected:
  // \brief Return the total number of elements in a list of component lists.
  static unsigned
  getComponentsTotalNumber(MappableExprComponentListsRef ComponentLists);

  // \brief Return the total number of elements in a list of declarations. All
  // declarations are expected to be canonical.
  static unsigned
  getUniqueDeclarationsTotalNumber(ArrayRef<ValueDecl *> Declarations);
};

/// \brief This represents clauses with a list of expressions that are mappable.
/// Examples of these clauses are 'map' in
/// '#pragma omp target [enter|exit] [data]...' directives, and  'to' and 'from
/// in '#pragma omp target update...' directives.
template <class T>
class OMPMappableExprListClause : public OMPVarListClause<T>,
                                  public OMPClauseMappableExprCommon {
  friend class OMPClauseReader;

  /// \brief Number of unique declarations in this clause.
  unsigned NumUniqueDeclarations;

  /// \brief Number of component lists in this clause.
  unsigned NumComponentLists;

  /// \brief Total number of components in this clause.
  unsigned NumComponents;

protected:
  /// \brief Get the unique declarations that are in the trailing objects of the
  /// class.
  MutableArrayRef<ValueDecl *> getUniqueDeclsRef() {
    return MutableArrayRef<ValueDecl *>(
        static_cast<T *>(this)->template getTrailingObjects<ValueDecl *>(),
        NumUniqueDeclarations);
  }

  /// \brief Get the unique declarations that are in the trailing objects of the
  /// class.
  ArrayRef<ValueDecl *> getUniqueDeclsRef() const {
    return ArrayRef<ValueDecl *>(
        static_cast<const T *>(this)
            ->template getTrailingObjects<ValueDecl *>(),
        NumUniqueDeclarations);
  }

  /// \brief Set the unique declarations that are in the trailing objects of the
  /// class.
  void setUniqueDecls(ArrayRef<ValueDecl *> UDs) {
    assert(UDs.size() == NumUniqueDeclarations &&
           "Unexpected amount of unique declarations.");
    std::copy(UDs.begin(), UDs.end(), getUniqueDeclsRef().begin());
  }

  /// \brief Get the number of lists per declaration that are in the trailing
  /// objects of the class.
  MutableArrayRef<unsigned> getDeclNumListsRef() {
    return MutableArrayRef<unsigned>(
        static_cast<T *>(this)->template getTrailingObjects<unsigned>(),
        NumUniqueDeclarations);
  }

  /// \brief Get the number of lists per declaration that are in the trailing
  /// objects of the class.
  ArrayRef<unsigned> getDeclNumListsRef() const {
    return ArrayRef<unsigned>(
        static_cast<const T *>(this)->template getTrailingObjects<unsigned>(),
        NumUniqueDeclarations);
  }

  /// \brief Set the number of lists per declaration that are in the trailing
  /// objects of the class.
  void setDeclNumLists(ArrayRef<unsigned> DNLs) {
    assert(DNLs.size() == NumUniqueDeclarations &&
           "Unexpected amount of list numbers.");
    std::copy(DNLs.begin(), DNLs.end(), getDeclNumListsRef().begin());
  }

  /// \brief Get the cumulative component lists sizes that are in the trailing
  /// objects of the class. They are appended after the number of lists.
  MutableArrayRef<unsigned> getComponentListSizesRef() {
    return MutableArrayRef<unsigned>(
        static_cast<T *>(this)->template getTrailingObjects<unsigned>() +
            NumUniqueDeclarations,
        NumComponentLists);
  }

  /// \brief Get the cumulative component lists sizes that are in the trailing
  /// objects of the class. They are appended after the number of lists.
  ArrayRef<unsigned> getComponentListSizesRef() const {
    return ArrayRef<unsigned>(
        static_cast<const T *>(this)->template getTrailingObjects<unsigned>() +
            NumUniqueDeclarations,
        NumComponentLists);
  }

  /// \brief Set the cumulative component lists sizes that are in the trailing
  /// objects of the class.
  void setComponentListSizes(ArrayRef<unsigned> CLSs) {
    assert(CLSs.size() == NumComponentLists &&
           "Unexpected amount of component lists.");
    std::copy(CLSs.begin(), CLSs.end(), getComponentListSizesRef().begin());
  }

  /// \brief Get the components that are in the trailing objects of the class.
  MutableArrayRef<MappableComponent> getComponentsRef() {
    return MutableArrayRef<MappableComponent>(
        static_cast<T *>(this)
            ->template getTrailingObjects<MappableComponent>(),
        NumComponents);
  }

  /// \brief Get the components that are in the trailing objects of the class.
  ArrayRef<MappableComponent> getComponentsRef() const {
    return ArrayRef<MappableComponent>(
        static_cast<const T *>(this)
            ->template getTrailingObjects<MappableComponent>(),
        NumComponents);
  }

  /// \brief Set the components that are in the trailing objects of the class.
  /// This requires the list sizes so that it can also fill the original
  /// expressions, which are the first component of each list.
  void setComponents(ArrayRef<MappableComponent> Components,
                     ArrayRef<unsigned> CLSs) {
    assert(Components.size() == NumComponents &&
           "Unexpected amount of component lists.");
    assert(CLSs.size() == NumComponentLists &&
           "Unexpected amount of list sizes.");
    std::copy(Components.begin(), Components.end(), getComponentsRef().begin());
  }

  /// \brief Fill the clause information from the list of declarations and
  /// associated component lists.
  void setClauseInfo(ArrayRef<ValueDecl *> Declarations,
                     MappableExprComponentListsRef ComponentLists) {
    // Perform some checks to make sure the data sizes are consistent with the
    // information available when the clause was created.
    assert(getUniqueDeclarationsTotalNumber(Declarations) ==
               NumUniqueDeclarations &&
           "Unexpected number of mappable expression info entries!");
    assert(getComponentsTotalNumber(ComponentLists) == NumComponents &&
           "Unexpected total number of components!");
    assert(Declarations.size() == ComponentLists.size() &&
           "Declaration and component lists size is not consistent!");
    assert(Declarations.size() == NumComponentLists &&
           "Unexpected declaration and component lists size!");

    // Organize the components by declaration and retrieve the original
    // expression. Original expressions are always the first component of the
    // mappable component list.
    llvm::DenseMap<ValueDecl *, SmallVector<MappableExprComponentListRef, 8>>
        ComponentListMap;
    {
      auto CI = ComponentLists.begin();
      for (auto DI = Declarations.begin(), DE = Declarations.end(); DI != DE;
           ++DI, ++CI) {
        assert(!CI->empty() && "Invalid component list!");
        ComponentListMap[*DI].push_back(*CI);
      }
    }

    // Iterators of the target storage.
    auto UniqueDeclarations = getUniqueDeclsRef();
    auto UDI = UniqueDeclarations.begin();

    auto DeclNumLists = getDeclNumListsRef();
    auto DNLI = DeclNumLists.begin();

    auto ComponentListSizes = getComponentListSizesRef();
    auto CLSI = ComponentListSizes.begin();

    auto Components = getComponentsRef();
    auto CI = Components.begin();

    // Variable to compute the accumulation of the number of components.
    unsigned PrevSize = 0u;

    // Scan all the declarations and associated component lists.
    for (auto &M : ComponentListMap) {
      // The declaration.
      auto *D = M.first;
      // The component lists.
      auto CL = M.second;

      // Initialize the entry.
      *UDI = D;
      ++UDI;

      *DNLI = CL.size();
      ++DNLI;

      // Obtain the cumulative sizes and concatenate all the components in the
      // reserved storage.
      for (auto C : CL) {
        // Accumulate with the previous size.
        PrevSize += C.size();

        // Save the size.
        *CLSI = PrevSize;
        ++CLSI;

        // Append components after the current components iterator.
        CI = std::copy(C.begin(), C.end(), CI);
      }
    }
  }

  /// \brief Build a clause for \a NumUniqueDeclarations declarations, \a
  /// NumComponentLists total component lists, and \a NumComponents total
  /// components.
  ///
  /// \param K Kind of the clause.
  /// \param StartLoc Starting location of the clause (the clause keyword).
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause - one
  /// list for each expression in the clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  OMPMappableExprListClause(OpenMPClauseKind K, SourceLocation StartLoc,
                            SourceLocation LParenLoc, SourceLocation EndLoc,
                            unsigned NumVars, unsigned NumUniqueDeclarations,
                            unsigned NumComponentLists, unsigned NumComponents)
      : OMPVarListClause<T>(K, StartLoc, LParenLoc, EndLoc, NumVars),
        NumUniqueDeclarations(NumUniqueDeclarations),
        NumComponentLists(NumComponentLists), NumComponents(NumComponents) {}

public:
  /// \brief Return the number of unique base declarations in this clause.
  unsigned getUniqueDeclarationsNum() const { return NumUniqueDeclarations; }
  /// \brief Return the number of lists derived from the clause expressions.
  unsigned getTotalComponentListNum() const { return NumComponentLists; }
  /// \brief Return the total number of components in all lists derived from the
  /// clause.
  unsigned getTotalComponentsNum() const { return NumComponents; }

  /// \brief Iterator that browse the components by lists. It also allows
  /// browsing components of a single declaration.
  class const_component_lists_iterator
      : public llvm::iterator_adaptor_base<
            const_component_lists_iterator,
            MappableExprComponentListRef::const_iterator,
            std::forward_iterator_tag, MappableComponent, ptrdiff_t,
            MappableComponent, MappableComponent> {
    // The declaration the iterator currently refers to.
    ArrayRef<ValueDecl *>::iterator DeclCur;

    // The list number associated with the current declaration.
    ArrayRef<unsigned>::iterator NumListsCur;

    // Remaining lists for the current declaration.
    unsigned RemainingLists;

    // The cumulative size of the previous list, or zero if there is no previous
    // list.
    unsigned PrevListSize;

    // The cumulative sizes of the current list - it will delimit the remaining
    // range of interest.
    ArrayRef<unsigned>::const_iterator ListSizeCur;
    ArrayRef<unsigned>::const_iterator ListSizeEnd;

    // Iterator to the end of the components storage.
    MappableExprComponentListRef::const_iterator End;

  public:
    /// \brief Construct an iterator that scans all lists.
    explicit const_component_lists_iterator(
        ArrayRef<ValueDecl *> UniqueDecls, ArrayRef<unsigned> DeclsListNum,
        ArrayRef<unsigned> CumulativeListSizes,
        MappableExprComponentListRef Components)
        : const_component_lists_iterator::iterator_adaptor_base(
              Components.begin()),
          DeclCur(UniqueDecls.begin()), NumListsCur(DeclsListNum.begin()),
          RemainingLists(0u), PrevListSize(0u),
          ListSizeCur(CumulativeListSizes.begin()),
          ListSizeEnd(CumulativeListSizes.end()), End(Components.end()) {
      assert(UniqueDecls.size() == DeclsListNum.size() &&
             "Inconsistent number of declarations and list sizes!");
      if (!DeclsListNum.empty())
        RemainingLists = *NumListsCur;
    }

    /// \brief Construct an iterator that scan lists for a given declaration \a
    /// Declaration.
    explicit const_component_lists_iterator(
        const ValueDecl *Declaration, ArrayRef<ValueDecl *> UniqueDecls,
        ArrayRef<unsigned> DeclsListNum, ArrayRef<unsigned> CumulativeListSizes,
        MappableExprComponentListRef Components)
        : const_component_lists_iterator(UniqueDecls, DeclsListNum,
                                         CumulativeListSizes, Components) {

      // Look for the desired declaration. While we are looking for it, we
      // update the state so that we know the component where a given list
      // starts.
      for (; DeclCur != UniqueDecls.end(); ++DeclCur, ++NumListsCur) {
        if (*DeclCur == Declaration)
          break;

        assert(*NumListsCur > 0 && "No lists associated with declaration??");

        // Skip the lists associated with the current declaration, but save the
        // last list size that was skipped.
        std::advance(ListSizeCur, *NumListsCur - 1);
        PrevListSize = *ListSizeCur;
        ++ListSizeCur;
      }

      // If we didn't find any declaration, advance the iterator to after the
      // last component and set remaining lists to zero.
      if (ListSizeCur == CumulativeListSizes.end()) {
        this->I = End;
        RemainingLists = 0u;
        return;
      }

      // Set the remaining lists with the total number of lists of the current
      // declaration.
      RemainingLists = *NumListsCur;

      // Adjust the list size end iterator to the end of the relevant range.
      ListSizeEnd = ListSizeCur;
      std::advance(ListSizeEnd, RemainingLists);

      // Given that the list sizes are cumulative, the index of the component
      // that start the list is the size of the previous list.
      std::advance(this->I, PrevListSize);
    }

    // Return the array with the current list. The sizes are cumulative, so the
    // array size is the difference between the current size and previous one.
    std::pair<const ValueDecl *, MappableExprComponentListRef>
    operator*() const {
      assert(ListSizeCur != ListSizeEnd && "Invalid iterator!");
      return std::make_pair(
          *DeclCur,
          MappableExprComponentListRef(&*this->I, *ListSizeCur - PrevListSize));
    }
    std::pair<const ValueDecl *, MappableExprComponentListRef>
    operator->() const {
      return **this;
    }

    // Skip the components of the current list.
    const_component_lists_iterator &operator++() {
      assert(ListSizeCur != ListSizeEnd && RemainingLists &&
             "Invalid iterator!");

      // If we don't have more lists just skip all the components. Otherwise,
      // advance the iterator by the number of components in the current list.
      if (std::next(ListSizeCur) == ListSizeEnd) {
        this->I = End;
        RemainingLists = 0;
      } else {
        std::advance(this->I, *ListSizeCur - PrevListSize);
        PrevListSize = *ListSizeCur;

        // We are done with a declaration, move to the next one.
        if (!(--RemainingLists)) {
          ++DeclCur;
          ++NumListsCur;
          RemainingLists = *NumListsCur;
          assert(RemainingLists && "No lists in the following declaration??");
        }
      }

      ++ListSizeCur;
      return *this;
    }
  };

  typedef llvm::iterator_range<const_component_lists_iterator>
      const_component_lists_range;

  /// \brief Iterators for all component lists.
  const_component_lists_iterator component_lists_begin() const {
    return const_component_lists_iterator(
        getUniqueDeclsRef(), getDeclNumListsRef(), getComponentListSizesRef(),
        getComponentsRef());
  }
  const_component_lists_iterator component_lists_end() const {
    return const_component_lists_iterator(
        ArrayRef<ValueDecl *>(), ArrayRef<unsigned>(), ArrayRef<unsigned>(),
        MappableExprComponentListRef(getComponentsRef().end(),
                                     getComponentsRef().end()));
  }
  const_component_lists_range component_lists() const {
    return {component_lists_begin(), component_lists_end()};
  }

  /// \brief Iterators for component lists associated with the provided
  /// declaration.
  const_component_lists_iterator
  decl_component_lists_begin(const ValueDecl *VD) const {
    return const_component_lists_iterator(
        VD, getUniqueDeclsRef(), getDeclNumListsRef(),
        getComponentListSizesRef(), getComponentsRef());
  }
  const_component_lists_iterator decl_component_lists_end() const {
    return component_lists_end();
  }
  const_component_lists_range decl_component_lists(const ValueDecl *VD) const {
    return {decl_component_lists_begin(VD), decl_component_lists_end()};
  }

  /// Iterators to access all the declarations, number of lists, list sizes, and
  /// components.
  typedef ArrayRef<ValueDecl *>::iterator const_all_decls_iterator;
  typedef llvm::iterator_range<const_all_decls_iterator> const_all_decls_range;
  const_all_decls_range all_decls() const {
    auto A = getUniqueDeclsRef();
    return const_all_decls_range(A.begin(), A.end());
  }

  typedef ArrayRef<unsigned>::iterator const_all_num_lists_iterator;
  typedef llvm::iterator_range<const_all_num_lists_iterator>
      const_all_num_lists_range;
  const_all_num_lists_range all_num_lists() const {
    auto A = getDeclNumListsRef();
    return const_all_num_lists_range(A.begin(), A.end());
  }

  typedef ArrayRef<unsigned>::iterator const_all_lists_sizes_iterator;
  typedef llvm::iterator_range<const_all_lists_sizes_iterator>
      const_all_lists_sizes_range;
  const_all_lists_sizes_range all_lists_sizes() const {
    auto A = getComponentListSizesRef();
    return const_all_lists_sizes_range(A.begin(), A.end());
  }

  typedef ArrayRef<MappableComponent>::iterator const_all_components_iterator;
  typedef llvm::iterator_range<const_all_components_iterator>
      const_all_components_range;
  const_all_components_range all_components() const {
    auto A = getComponentsRef();
    return const_all_components_range(A.begin(), A.end());
  }
};

/// \brief This represents clause 'map' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp target map(a,b)
/// \endcode
/// In this example directive '#pragma omp target' has clause 'map'
/// with the variables 'a' and 'b'.
///
class OMPMapClause final : public OMPMappableExprListClause<OMPMapClause>,
                           private llvm::TrailingObjects<
                               OMPMapClause, Expr *, ValueDecl *, unsigned,
                               OMPClauseMappableExprCommon::MappableComponent> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend OMPMappableExprListClause;
  friend class OMPClauseReader;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return varlist_size();
  }
  size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
    return getUniqueDeclarationsNum();
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const {
    return getUniqueDeclarationsNum() + getTotalComponentListNum();
  }

  /// \brief Map type modifier for the 'map' clause.
  OpenMPMapClauseKind MapTypeModifier;
  /// \brief Map type for the 'map' clause.
  OpenMPMapClauseKind MapType;
  /// \brief Is this an implicit map type or not.
  bool MapTypeIsImplicit;
  /// \brief Location of the map type.
  SourceLocation MapLoc;
  /// \brief Colon location.
  SourceLocation ColonLoc;

  /// \brief Set type modifier for the clause.
  ///
  /// \param T Type Modifier for the clause.
  ///
  void setMapTypeModifier(OpenMPMapClauseKind T) { MapTypeModifier = T; }

  /// \brief Set type for the clause.
  ///
  /// \param T Type for the clause.
  ///
  void setMapType(OpenMPMapClauseKind T) { MapType = T; }

  /// \brief Set type location.
  ///
  /// \param TLoc Type location.
  ///
  void setMapLoc(SourceLocation TLoc) { MapLoc = TLoc; }

  /// \brief Set colon location.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

  /// \brief Build a clause for \a NumVars listed expressions, \a
  /// NumUniqueDeclarations declarations, \a NumComponentLists total component
  /// lists, and \a NumComponents total expression components.
  ///
  /// \param MapTypeModifier Map type modifier.
  /// \param MapType Map type.
  /// \param MapTypeIsImplicit Map type is inferred implicitly.
  /// \param MapLoc Location of the map type.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPMapClause(OpenMPMapClauseKind MapTypeModifier,
                        OpenMPMapClauseKind MapType, bool MapTypeIsImplicit,
                        SourceLocation MapLoc, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        unsigned NumVars, unsigned NumUniqueDeclarations,
                        unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_map, StartLoc, LParenLoc, EndLoc,
                                  NumVars, NumUniqueDeclarations,
                                  NumComponentLists, NumComponents),
        MapTypeModifier(MapTypeModifier), MapType(MapType),
        MapTypeIsImplicit(MapTypeIsImplicit), MapLoc(MapLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPMapClause(unsigned NumVars, unsigned NumUniqueDeclarations,
                        unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(
            OMPC_map, SourceLocation(), SourceLocation(), SourceLocation(),
            NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents),
        MapTypeModifier(OMPC_MAP_unknown), MapType(OMPC_MAP_unknown),
        MapTypeIsImplicit(false), MapLoc() {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param Vars The original expression used in the clause.
  /// \param Declarations Declarations used in the clause.
  /// \param ComponentLists Component lists used in the clause.
  /// \param TypeModifier Map type modifier.
  /// \param Type Map type.
  /// \param TypeIsImplicit Map type is inferred implicitly.
  /// \param TypeLoc Location of the map type.
  ///
  static OMPMapClause *Create(const ASTContext &C, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc,
                              ArrayRef<Expr *> Vars,
                              ArrayRef<ValueDecl *> Declarations,
                              MappableExprComponentListsRef ComponentLists,
                              OpenMPMapClauseKind TypeModifier,
                              OpenMPMapClauseKind Type, bool TypeIsImplicit,
                              SourceLocation TypeLoc);
  /// \brief Creates an empty clause with the place for for \a NumVars original
  /// expressions, \a NumUniqueDeclarations declarations, \NumComponentLists
  /// lists, and \a NumComponents expression components.
  ///
  /// \param C AST context.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of unique base declarations in this
  /// clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  static OMPMapClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
                                   unsigned NumUniqueDeclarations,
                                   unsigned NumComponentLists,
                                   unsigned NumComponents);

  /// \brief Fetches mapping kind for the clause.
  OpenMPMapClauseKind getMapType() const LLVM_READONLY { return MapType; }

  /// \brief Is this an implicit map type?
  /// We have to capture 'IsMapTypeImplicit' from the parser for more
  /// informative error messages.  It helps distinguish map(r) from
  /// map(tofrom: r), which is important to print more helpful error
  /// messages for some target directives.
  bool isImplicitMapType() const LLVM_READONLY { return MapTypeIsImplicit; }

  /// \brief Fetches the map type modifier for the clause.
  OpenMPMapClauseKind getMapTypeModifier() const LLVM_READONLY {
    return MapTypeModifier;
  }

  /// \brief Fetches location of clause mapping kind.
  SourceLocation getMapLoc() const LLVM_READONLY { return MapLoc; }

  /// \brief Get colon location.
  SourceLocation getColonLoc() const { return ColonLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_map;
  }

  child_range children() {
    return child_range(
        reinterpret_cast<Stmt **>(varlist_begin()),
        reinterpret_cast<Stmt **>(varlist_end()));
  }
};

/// \brief This represents 'num_teams' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp teams num_teams(n)
/// \endcode
/// In this example directive '#pragma omp teams' has clause 'num_teams'
/// with single expression 'n'.
///
class OMPNumTeamsClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief NumTeams number.
  Stmt *NumTeams;
  /// \brief Set the NumTeams number.
  ///
  /// \param E NumTeams number.
  ///
  void setNumTeams(Expr *E) { NumTeams = E; }

public:
  /// \brief Build 'num_teams' clause.
  ///
  /// \param E Expression associated with this clause.
  /// \param HelperE Helper Expression associated with this clause.
  /// \param CaptureRegion Innermost OpenMP region where expressions in this
  /// clause must be captured.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNumTeamsClause(Expr *E, Stmt *HelperE, OpenMPDirectiveKind CaptureRegion,
                    SourceLocation StartLoc, SourceLocation LParenLoc,
                    SourceLocation EndLoc)
      : OMPClause(OMPC_num_teams, StartLoc, EndLoc), OMPClauseWithPreInit(this),
        LParenLoc(LParenLoc), NumTeams(E) {
    setPreInitStmt(HelperE, CaptureRegion);
  }

  /// \brief Build an empty clause.
  ///
  OMPNumTeamsClause()
      : OMPClause(OMPC_num_teams, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), LParenLoc(SourceLocation()),
        NumTeams(nullptr) {}
  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }
  /// \brief Return NumTeams number.
  Expr *getNumTeams() { return cast<Expr>(NumTeams); }
  /// \brief Return NumTeams number.
  Expr *getNumTeams() const { return cast<Expr>(NumTeams); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_num_teams;
  }

  child_range children() { return child_range(&NumTeams, &NumTeams + 1); }
};

/// \brief This represents 'thread_limit' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp teams thread_limit(n)
/// \endcode
/// In this example directive '#pragma omp teams' has clause 'thread_limit'
/// with single expression 'n'.
///
class OMPThreadLimitClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief ThreadLimit number.
  Stmt *ThreadLimit;
  /// \brief Set the ThreadLimit number.
  ///
  /// \param E ThreadLimit number.
  ///
  void setThreadLimit(Expr *E) { ThreadLimit = E; }

public:
  /// \brief Build 'thread_limit' clause.
  ///
  /// \param E Expression associated with this clause.
  /// \param HelperE Helper Expression associated with this clause.
  /// \param CaptureRegion Innermost OpenMP region where expressions in this
  /// clause must be captured.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPThreadLimitClause(Expr *E, Stmt *HelperE,
                       OpenMPDirectiveKind CaptureRegion,
                       SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation EndLoc)
      : OMPClause(OMPC_thread_limit, StartLoc, EndLoc),
        OMPClauseWithPreInit(this), LParenLoc(LParenLoc), ThreadLimit(E) {
    setPreInitStmt(HelperE, CaptureRegion);
  }

  /// \brief Build an empty clause.
  ///
  OMPThreadLimitClause()
      : OMPClause(OMPC_thread_limit, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), LParenLoc(SourceLocation()),
        ThreadLimit(nullptr) {}
  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }
  /// \brief Return ThreadLimit number.
  Expr *getThreadLimit() { return cast<Expr>(ThreadLimit); }
  /// \brief Return ThreadLimit number.
  Expr *getThreadLimit() const { return cast<Expr>(ThreadLimit); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_thread_limit;
  }

  child_range children() { return child_range(&ThreadLimit, &ThreadLimit + 1); }
};

/// \brief This represents 'priority' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp task priority(n)
/// \endcode
/// In this example directive '#pragma omp teams' has clause 'priority' with
/// single expression 'n'.
///
class OMPPriorityClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Priority number.
  Stmt *Priority;
  /// \brief Set the Priority number.
  ///
  /// \param E Priority number.
  ///
  void setPriority(Expr *E) { Priority = E; }

public:
  /// \brief Build 'priority' clause.
  ///
  /// \param E Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPPriorityClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                    SourceLocation EndLoc)
      : OMPClause(OMPC_priority, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Priority(E) {}

  /// \brief Build an empty clause.
  ///
  OMPPriorityClause()
      : OMPClause(OMPC_priority, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Priority(nullptr) {}
  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }
  /// \brief Return Priority number.
  Expr *getPriority() { return cast<Expr>(Priority); }
  /// \brief Return Priority number.
  Expr *getPriority() const { return cast<Expr>(Priority); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_priority;
  }

  child_range children() { return child_range(&Priority, &Priority + 1); }
};

/// \brief This represents 'grainsize' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp taskloop grainsize(4)
/// \endcode
/// In this example directive '#pragma omp taskloop' has clause 'grainsize'
/// with single expression '4'.
///
class OMPGrainsizeClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Safe iteration space distance.
  Stmt *Grainsize;

  /// \brief Set safelen.
  void setGrainsize(Expr *Size) { Grainsize = Size; }

public:
  /// \brief Build 'grainsize' clause.
  ///
  /// \param Size Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPGrainsizeClause(Expr *Size, SourceLocation StartLoc,
                     SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_grainsize, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Grainsize(Size) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPGrainsizeClause()
      : OMPClause(OMPC_grainsize, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Grainsize(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return safe iteration space distance.
  Expr *getGrainsize() const { return cast_or_null<Expr>(Grainsize); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_grainsize;
  }

  child_range children() { return child_range(&Grainsize, &Grainsize + 1); }
};

/// \brief This represents 'nogroup' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp taskloop nogroup
/// \endcode
/// In this example directive '#pragma omp taskloop' has 'nogroup' clause.
///
class OMPNogroupClause : public OMPClause {
public:
  /// \brief Build 'nogroup' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNogroupClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_nogroup, StartLoc, EndLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPNogroupClause()
      : OMPClause(OMPC_nogroup, SourceLocation(), SourceLocation()) {}

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_nogroup;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents 'num_tasks' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp taskloop num_tasks(4)
/// \endcode
/// In this example directive '#pragma omp taskloop' has clause 'num_tasks'
/// with single expression '4'.
///
class OMPNumTasksClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Safe iteration space distance.
  Stmt *NumTasks;

  /// \brief Set safelen.
  void setNumTasks(Expr *Size) { NumTasks = Size; }

public:
  /// \brief Build 'num_tasks' clause.
  ///
  /// \param Size Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNumTasksClause(Expr *Size, SourceLocation StartLoc,
                    SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_num_tasks, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumTasks(Size) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPNumTasksClause()
      : OMPClause(OMPC_num_tasks, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumTasks(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return safe iteration space distance.
  Expr *getNumTasks() const { return cast_or_null<Expr>(NumTasks); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_num_tasks;
  }

  child_range children() { return child_range(&NumTasks, &NumTasks + 1); }
};

/// \brief This represents 'hint' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp critical (name) hint(6)
/// \endcode
/// In this example directive '#pragma omp critical' has name 'name' and clause
/// 'hint' with argument '6'.
///
class OMPHintClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Hint expression of the 'hint' clause.
  Stmt *Hint;

  /// \brief Set hint expression.
  ///
  void setHint(Expr *H) { Hint = H; }

public:
  /// \brief Build 'hint' clause with expression \a Hint.
  ///
  /// \param Hint Hint expression.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPHintClause(Expr *Hint, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OMPClause(OMPC_hint, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Hint(Hint) {}

  /// \brief Build an empty clause.
  ///
  OMPHintClause()
      : OMPClause(OMPC_hint, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Hint(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns number of threads.
  Expr *getHint() const { return cast_or_null<Expr>(Hint); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_hint;
  }

  child_range children() { return child_range(&Hint, &Hint + 1); }
};

/// \brief This represents 'dist_schedule' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp distribute dist_schedule(static, 3)
/// \endcode
/// In this example directive '#pragma omp distribute' has 'dist_schedule'
/// clause with arguments 'static' and '3'.
///
class OMPDistScheduleClause : public OMPClause, public OMPClauseWithPreInit {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'schedule' clause.
  OpenMPDistScheduleClauseKind Kind;
  /// \brief Start location of the schedule kind in source code.
  SourceLocation KindLoc;
  /// \brief Location of ',' (if any).
  SourceLocation CommaLoc;
  /// \brief Chunk size.
  Expr *ChunkSize;

  /// \brief Set schedule kind.
  ///
  /// \param K Schedule kind.
  ///
  void setDistScheduleKind(OpenMPDistScheduleClauseKind K) { Kind = K; }
  /// \brief Sets the location of '('.
  ///
  /// \param Loc Location of '('.
  ///
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Set schedule kind start location.
  ///
  /// \param KLoc Schedule kind location.
  ///
  void setDistScheduleKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }
  /// \brief Set location of ','.
  ///
  /// \param Loc Location of ','.
  ///
  void setCommaLoc(SourceLocation Loc) { CommaLoc = Loc; }
  /// \brief Set chunk size.
  ///
  /// \param E Chunk size.
  ///
  void setChunkSize(Expr *E) { ChunkSize = E; }

public:
  /// \brief Build 'dist_schedule' clause with schedule kind \a Kind and chunk
  /// size expression \a ChunkSize.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param KLoc Starting location of the argument.
  /// \param CommaLoc Location of ','.
  /// \param EndLoc Ending location of the clause.
  /// \param Kind DistSchedule kind.
  /// \param ChunkSize Chunk size.
  /// \param HelperChunkSize Helper chunk size for combined directives.
  ///
  OMPDistScheduleClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                        SourceLocation KLoc, SourceLocation CommaLoc,
                        SourceLocation EndLoc,
                        OpenMPDistScheduleClauseKind Kind, Expr *ChunkSize,
                        Stmt *HelperChunkSize)
      : OMPClause(OMPC_dist_schedule, StartLoc, EndLoc),
        OMPClauseWithPreInit(this), LParenLoc(LParenLoc), Kind(Kind),
        KindLoc(KLoc), CommaLoc(CommaLoc), ChunkSize(ChunkSize) {
    setPreInitStmt(HelperChunkSize);
  }

  /// \brief Build an empty clause.
  ///
  explicit OMPDistScheduleClause()
      : OMPClause(OMPC_dist_schedule, SourceLocation(), SourceLocation()),
        OMPClauseWithPreInit(this), Kind(OMPC_DIST_SCHEDULE_unknown),
        ChunkSize(nullptr) {}

  /// \brief Get kind of the clause.
  ///
  OpenMPDistScheduleClauseKind getDistScheduleKind() const { return Kind; }
  /// \brief Get location of '('.
  ///
  SourceLocation getLParenLoc() { return LParenLoc; }
  /// \brief Get kind location.
  ///
  SourceLocation getDistScheduleKindLoc() { return KindLoc; }
  /// \brief Get location of ','.
  ///
  SourceLocation getCommaLoc() { return CommaLoc; }
  /// \brief Get chunk size.
  ///
  Expr *getChunkSize() { return ChunkSize; }
  /// \brief Get chunk size.
  ///
  const Expr *getChunkSize() const { return ChunkSize; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_dist_schedule;
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(&ChunkSize),
                       reinterpret_cast<Stmt **>(&ChunkSize) + 1);
  }
};

/// \brief This represents 'defaultmap' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp target defaultmap(tofrom: scalar)
/// \endcode
/// In this example directive '#pragma omp target' has 'defaultmap' clause of kind
/// 'scalar' with modifier 'tofrom'.
///
class OMPDefaultmapClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Modifiers for 'defaultmap' clause.
  OpenMPDefaultmapClauseModifier Modifier;
  /// \brief Locations of modifiers.
  SourceLocation ModifierLoc;
  /// \brief A kind of the 'defaultmap' clause.
  OpenMPDefaultmapClauseKind Kind;
  /// \brief Start location of the defaultmap kind in source code.
  SourceLocation KindLoc;

  /// \brief Set defaultmap kind.
  ///
  /// \param K Defaultmap kind.
  ///
  void setDefaultmapKind(OpenMPDefaultmapClauseKind K) { Kind = K; }
  /// \brief Set the defaultmap modifier.
  ///
  /// \param M Defaultmap modifier.
  ///
  void setDefaultmapModifier(OpenMPDefaultmapClauseModifier M) {
    Modifier = M;
  }
  /// \brief Set location of the defaultmap modifier.
  ///
  void setDefaultmapModifierLoc(SourceLocation Loc) {
    ModifierLoc = Loc;
  }
  /// \brief Sets the location of '('.
  ///
  /// \param Loc Location of '('.
  ///
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Set defaultmap kind start location.
  ///
  /// \param KLoc Defaultmap kind location.
  ///
  void setDefaultmapKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }

public:
  /// \brief Build 'defaultmap' clause with defaultmap kind \a Kind
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param KLoc Starting location of the argument.
  /// \param EndLoc Ending location of the clause.
  /// \param Kind Defaultmap kind.
  /// \param M The modifier applied to 'defaultmap' clause.
  /// \param MLoc Location of the modifier
  ///
  OMPDefaultmapClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                      SourceLocation MLoc, SourceLocation KLoc,
                      SourceLocation EndLoc, OpenMPDefaultmapClauseKind Kind,
                      OpenMPDefaultmapClauseModifier M)
      : OMPClause(OMPC_defaultmap, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Modifier(M), ModifierLoc(MLoc), Kind(Kind), KindLoc(KLoc) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPDefaultmapClause()
      : OMPClause(OMPC_defaultmap, SourceLocation(), SourceLocation()),
        Modifier(OMPC_DEFAULTMAP_MODIFIER_unknown),
        Kind(OMPC_DEFAULTMAP_unknown) {}

  /// \brief Get kind of the clause.
  ///
  OpenMPDefaultmapClauseKind getDefaultmapKind() const { return Kind; }
  /// \brief Get the modifier of the clause.
  ///
  OpenMPDefaultmapClauseModifier getDefaultmapModifier() const {
    return Modifier;
  }
  /// \brief Get location of '('.
  ///
  SourceLocation getLParenLoc() { return LParenLoc; }
  /// \brief Get kind location.
  ///
  SourceLocation getDefaultmapKindLoc() { return KindLoc; }
  /// \brief Get the modifier location.
  ///
  SourceLocation getDefaultmapModifierLoc() const {
    return ModifierLoc;
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_defaultmap;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// \brief This represents clause 'to' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp target update to(a,b)
/// \endcode
/// In this example directive '#pragma omp target update' has clause 'to'
/// with the variables 'a' and 'b'.
///
class OMPToClause final : public OMPMappableExprListClause<OMPToClause>,
                          private llvm::TrailingObjects<
                              OMPToClause, Expr *, ValueDecl *, unsigned,
                              OMPClauseMappableExprCommon::MappableComponent> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend OMPMappableExprListClause;
  friend class OMPClauseReader;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return varlist_size();
  }
  size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
    return getUniqueDeclarationsNum();
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const {
    return getUniqueDeclarationsNum() + getTotalComponentListNum();
  }

  /// \brief Build clause with number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPToClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation EndLoc, unsigned NumVars,
                       unsigned NumUniqueDeclarations,
                       unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_to, StartLoc, LParenLoc, EndLoc, NumVars,
                                  NumUniqueDeclarations, NumComponentLists,
                                  NumComponents) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPToClause(unsigned NumVars, unsigned NumUniqueDeclarations,
                       unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(
            OMPC_to, SourceLocation(), SourceLocation(), SourceLocation(),
            NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents) {}

public:
  /// \brief Creates clause with a list of variables \a Vars.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param Vars The original expression used in the clause.
  /// \param Declarations Declarations used in the clause.
  /// \param ComponentLists Component lists used in the clause.
  ///
  static OMPToClause *Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc,
                             ArrayRef<Expr *> Vars,
                             ArrayRef<ValueDecl *> Declarations,
                             MappableExprComponentListsRef ComponentLists);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of unique base declarations in this
  /// clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  static OMPToClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
                                  unsigned NumUniqueDeclarations,
                                  unsigned NumComponentLists,
                                  unsigned NumComponents);

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_to;
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }
};

/// \brief This represents clause 'from' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp target update from(a,b)
/// \endcode
/// In this example directive '#pragma omp target update' has clause 'from'
/// with the variables 'a' and 'b'.
///
class OMPFromClause final
    : public OMPMappableExprListClause<OMPFromClause>,
      private llvm::TrailingObjects<
          OMPFromClause, Expr *, ValueDecl *, unsigned,
          OMPClauseMappableExprCommon::MappableComponent> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend OMPMappableExprListClause;
  friend class OMPClauseReader;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return varlist_size();
  }
  size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
    return getUniqueDeclarationsNum();
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const {
    return getUniqueDeclarationsNum() + getTotalComponentListNum();
  }

  /// \brief Build clause with number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPFromClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                         SourceLocation EndLoc, unsigned NumVars,
                         unsigned NumUniqueDeclarations,
                         unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_from, StartLoc, LParenLoc, EndLoc,
                                  NumVars, NumUniqueDeclarations,
                                  NumComponentLists, NumComponents) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPFromClause(unsigned NumVars, unsigned NumUniqueDeclarations,
                         unsigned NumComponentLists, unsigned NumComponents)
      : OMPMappableExprListClause(
            OMPC_from, SourceLocation(), SourceLocation(), SourceLocation(),
            NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents) {}

public:
  /// \brief Creates clause with a list of variables \a Vars.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param Vars The original expression used in the clause.
  /// \param Declarations Declarations used in the clause.
  /// \param ComponentLists Component lists used in the clause.
  ///
  static OMPFromClause *Create(const ASTContext &C, SourceLocation StartLoc,
                               SourceLocation LParenLoc, SourceLocation EndLoc,
                               ArrayRef<Expr *> Vars,
                               ArrayRef<ValueDecl *> Declarations,
                               MappableExprComponentListsRef ComponentLists);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of unique base declarations in this
  /// clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  static OMPFromClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
                                    unsigned NumUniqueDeclarations,
                                    unsigned NumComponentLists,
                                    unsigned NumComponents);

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_from;
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }
};

/// This represents clause 'use_device_ptr' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp target data use_device_ptr(a,b)
/// \endcode
/// In this example directive '#pragma omp target data' has clause
/// 'use_device_ptr' with the variables 'a' and 'b'.
///
class OMPUseDevicePtrClause final
    : public OMPMappableExprListClause<OMPUseDevicePtrClause>,
      private llvm::TrailingObjects<
          OMPUseDevicePtrClause, Expr *, ValueDecl *, unsigned,
          OMPClauseMappableExprCommon::MappableComponent> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend OMPMappableExprListClause;
  friend class OMPClauseReader;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return 3 * varlist_size();
  }
  size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
    return getUniqueDeclarationsNum();
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const {
    return getUniqueDeclarationsNum() + getTotalComponentListNum();
  }

  /// Build clause with number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPUseDevicePtrClause(SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, unsigned NumVars,
                                 unsigned NumUniqueDeclarations,
                                 unsigned NumComponentLists,
                                 unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_use_device_ptr, StartLoc, LParenLoc,
                                  EndLoc, NumVars, NumUniqueDeclarations,
                                  NumComponentLists, NumComponents) {}

  /// Build an empty clause.
  ///
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPUseDevicePtrClause(unsigned NumVars,
                                 unsigned NumUniqueDeclarations,
                                 unsigned NumComponentLists,
                                 unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_use_device_ptr, SourceLocation(),
                                  SourceLocation(), SourceLocation(), NumVars,
                                  NumUniqueDeclarations, NumComponentLists,
                                  NumComponents) {}

  /// Sets the list of references to private copies with initializers for new
  /// private variables.
  /// \param VL List of references.
  void setPrivateCopies(ArrayRef<Expr *> VL);

  /// Gets the list of references to private copies with initializers for new
  /// private variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::makeArrayRef(varlist_end(), varlist_size());
  }

  /// Sets the list of references to initializer variables for new private
  /// variables.
  /// \param VL List of references.
  void setInits(ArrayRef<Expr *> VL);

  /// Gets the list of references to initializer variables for new private
  /// variables.
  MutableArrayRef<Expr *> getInits() {
    return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
  }
  ArrayRef<const Expr *> getInits() const {
    return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
  }

public:
  /// Creates clause with a list of variables \a Vars.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param Vars The original expression used in the clause.
  /// \param PrivateVars Expressions referring to private copies.
  /// \param Inits Expressions referring to private copy initializers.
  /// \param Declarations Declarations used in the clause.
  /// \param ComponentLists Component lists used in the clause.
  ///
  static OMPUseDevicePtrClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> Vars,
         ArrayRef<Expr *> PrivateVars, ArrayRef<Expr *> Inits,
         ArrayRef<ValueDecl *> Declarations,
         MappableExprComponentListsRef ComponentLists);

  /// Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of unique base declarations in this
  /// clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  static OMPUseDevicePtrClause *CreateEmpty(const ASTContext &C,
                                            unsigned NumVars,
                                            unsigned NumUniqueDeclarations,
                                            unsigned NumComponentLists,
                                            unsigned NumComponents);

  typedef MutableArrayRef<Expr *>::iterator private_copies_iterator;
  typedef ArrayRef<const Expr *>::iterator private_copies_const_iterator;
  typedef llvm::iterator_range<private_copies_iterator> private_copies_range;
  typedef llvm::iterator_range<private_copies_const_iterator>
      private_copies_const_range;

  private_copies_range private_copies() {
    return private_copies_range(getPrivateCopies().begin(),
                                getPrivateCopies().end());
  }
  private_copies_const_range private_copies() const {
    return private_copies_const_range(getPrivateCopies().begin(),
                                      getPrivateCopies().end());
  }

  typedef MutableArrayRef<Expr *>::iterator inits_iterator;
  typedef ArrayRef<const Expr *>::iterator inits_const_iterator;
  typedef llvm::iterator_range<inits_iterator> inits_range;
  typedef llvm::iterator_range<inits_const_iterator> inits_const_range;

  inits_range inits() {
    return inits_range(getInits().begin(), getInits().end());
  }
  inits_const_range inits() const {
    return inits_const_range(getInits().begin(), getInits().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_use_device_ptr;
  }
};

/// This represents clause 'is_device_ptr' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp target is_device_ptr(a,b)
/// \endcode
/// In this example directive '#pragma omp target' has clause
/// 'is_device_ptr' with the variables 'a' and 'b'.
///
class OMPIsDevicePtrClause final
    : public OMPMappableExprListClause<OMPIsDevicePtrClause>,
      private llvm::TrailingObjects<
          OMPIsDevicePtrClause, Expr *, ValueDecl *, unsigned,
          OMPClauseMappableExprCommon::MappableComponent> {
  friend TrailingObjects;
  friend OMPVarListClause;
  friend OMPMappableExprListClause;
  friend class OMPClauseReader;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return varlist_size();
  }
  size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
    return getUniqueDeclarationsNum();
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const {
    return getUniqueDeclarationsNum() + getTotalComponentListNum();
  }
  /// Build clause with number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPIsDevicePtrClause(SourceLocation StartLoc,
                                SourceLocation LParenLoc, SourceLocation EndLoc,
                                unsigned NumVars,
                                unsigned NumUniqueDeclarations,
                                unsigned NumComponentLists,
                                unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_is_device_ptr, StartLoc, LParenLoc,
                                  EndLoc, NumVars, NumUniqueDeclarations,
                                  NumComponentLists, NumComponents) {}

  /// Build an empty clause.
  ///
  /// \param NumVars Number of expressions listed in this clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of component lists in this clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  explicit OMPIsDevicePtrClause(unsigned NumVars,
                                unsigned NumUniqueDeclarations,
                                unsigned NumComponentLists,
                                unsigned NumComponents)
      : OMPMappableExprListClause(OMPC_is_device_ptr, SourceLocation(),
                                  SourceLocation(), SourceLocation(), NumVars,
                                  NumUniqueDeclarations, NumComponentLists,
                                  NumComponents) {}

public:
  /// Creates clause with a list of variables \a Vars.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  /// \param Vars The original expression used in the clause.
  /// \param Declarations Declarations used in the clause.
  /// \param ComponentLists Component lists used in the clause.
  ///
  static OMPIsDevicePtrClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> Vars,
         ArrayRef<ValueDecl *> Declarations,
         MappableExprComponentListsRef ComponentLists);

  /// Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of expressions listed in the clause.
  /// \param NumUniqueDeclarations Number of unique base declarations in this
  /// clause.
  /// \param NumComponentLists Number of unique base declarations in this
  /// clause.
  /// \param NumComponents Total number of expression components in the clause.
  ///
  static OMPIsDevicePtrClause *CreateEmpty(const ASTContext &C,
                                           unsigned NumVars,
                                           unsigned NumUniqueDeclarations,
                                           unsigned NumComponentLists,
                                           unsigned NumComponents);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_is_device_ptr;
  }
};
} // end namespace clang

#endif // LLVM_CLANG_AST_OPENMPCLAUSE_H
