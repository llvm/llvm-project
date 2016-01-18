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
        reinterpret_cast<Expr **>(
            reinterpret_cast<char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<Expr *>())),
        NumVars);
  }

  /// \brief Sets the list of variables for this clause.
  void setVarRefs(ArrayRef<Expr *> VL) {
    assert(VL.size() == NumVars &&
           "Number of variables is not the same as the preallocated buffer");
    std::copy(
        VL.begin(), VL.end(),
        reinterpret_cast<Expr **>(
            reinterpret_cast<char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<Expr *>())));
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
        reinterpret_cast<const Expr *const *>(
            reinterpret_cast<const char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<const Expr *>())),
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
class OMPIfClause : public OMPClause {
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
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param NameModifierLoc Location of directive name modifier.
  /// \param ColonLoc [OpenMP 4.1] Location of ':'.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPIfClause(OpenMPDirectiveKind NameModifier, Expr *Cond,
              SourceLocation StartLoc, SourceLocation LParenLoc,
              SourceLocation NameModifierLoc, SourceLocation ColonLoc,
              SourceLocation EndLoc)
      : OMPClause(OMPC_if, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond), ColonLoc(ColonLoc), NameModifier(NameModifier),
        NameModifierLoc(NameModifierLoc) {}

  /// \brief Build an empty clause.
  ///
  OMPIfClause()
      : OMPClause(OMPC_if, SourceLocation(), SourceLocation()), LParenLoc(),
        Condition(nullptr), ColonLoc(), NameModifier(OMPD_unknown),
        NameModifierLoc() {}

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
class OMPNumThreadsClause : public OMPClause {
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
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNumThreadsClause(Expr *NumThreads, SourceLocation StartLoc,
                      SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_num_threads, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumThreads(NumThreads) {}

  /// \brief Build an empty clause.
  ///
  OMPNumThreadsClause()
      : OMPClause(OMPC_num_threads, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumThreads(nullptr) {}

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
class OMPScheduleClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'schedule' clause.
  OpenMPScheduleClauseKind Kind;
  /// \brief Start location of the schedule ind in source code.
  SourceLocation KindLoc;
  /// \brief Location of ',' (if any).
  SourceLocation CommaLoc;
  /// \brief Chunk size and a reference to pseudo variable for combined
  /// directives.
  enum { CHUNK_SIZE, HELPER_CHUNK_SIZE, NUM_EXPRS };
  Stmt *ChunkSizes[NUM_EXPRS];

  /// \brief Set schedule kind.
  ///
  /// \param K Schedule kind.
  ///
  void setScheduleKind(OpenMPScheduleClauseKind K) { Kind = K; }
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
  void setChunkSize(Expr *E) { ChunkSizes[CHUNK_SIZE] = E; }
  /// \brief Set helper chunk size.
  ///
  /// \param E Helper chunk size.
  ///
  void setHelperChunkSize(Expr *E) { ChunkSizes[HELPER_CHUNK_SIZE] = E; }

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
  ///
  OMPScheduleClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                    SourceLocation KLoc, SourceLocation CommaLoc,
                    SourceLocation EndLoc, OpenMPScheduleClauseKind Kind,
                    Expr *ChunkSize, Expr *HelperChunkSize)
      : OMPClause(OMPC_schedule, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(Kind), KindLoc(KLoc), CommaLoc(CommaLoc) {
    ChunkSizes[CHUNK_SIZE] = ChunkSize;
    ChunkSizes[HELPER_CHUNK_SIZE] = HelperChunkSize;
  }

  /// \brief Build an empty clause.
  ///
  explicit OMPScheduleClause()
      : OMPClause(OMPC_schedule, SourceLocation(), SourceLocation()),
        Kind(OMPC_SCHEDULE_unknown) {
    ChunkSizes[CHUNK_SIZE] = nullptr;
    ChunkSizes[HELPER_CHUNK_SIZE] = nullptr;
  }

  /// \brief Get kind of the clause.
  ///
  OpenMPScheduleClauseKind getScheduleKind() const { return Kind; }
  /// \brief Get location of '('.
  ///
  SourceLocation getLParenLoc() { return LParenLoc; }
  /// \brief Get kind location.
  ///
  SourceLocation getScheduleKindLoc() { return KindLoc; }
  /// \brief Get location of ','.
  ///
  SourceLocation getCommaLoc() { return CommaLoc; }
  /// \brief Get chunk size.
  ///
  Expr *getChunkSize() { return dyn_cast_or_null<Expr>(ChunkSizes[CHUNK_SIZE]); }
  /// \brief Get chunk size.
  ///
  Expr *getChunkSize() const {
    return dyn_cast_or_null<Expr>(ChunkSizes[CHUNK_SIZE]);
  }
  /// \brief Get helper chunk size.
  ///
  Expr *getHelperChunkSize() {
    return dyn_cast_or_null<Expr>(ChunkSizes[HELPER_CHUNK_SIZE]);
  }
  /// \brief Get helper chunk size.
  ///
  Expr *getHelperChunkSize() const {
    return dyn_cast_or_null<Expr>(ChunkSizes[HELPER_CHUNK_SIZE]);
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_schedule;
  }

  child_range children() {
    return child_range(&ChunkSizes[CHUNK_SIZE], &ChunkSizes[CHUNK_SIZE] + 1);
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
class OMPPrivateClause : public OMPVarListClause<OMPPrivateClause> {
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
class OMPFirstprivateClause : public OMPVarListClause<OMPFirstprivateClause> {
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
                                                LParenLoc, EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPFirstprivateClause(unsigned N)
      : OMPVarListClause<OMPFirstprivateClause>(
            OMPC_firstprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N) {}
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
  ///
  static OMPFirstprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
         ArrayRef<Expr *> InitVL);
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
class OMPLastprivateClause : public OMPVarListClause<OMPLastprivateClause> {
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
                                               LParenLoc, EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPLastprivateClause(unsigned N)
      : OMPVarListClause<OMPLastprivateClause>(
            OMPC_lastprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N) {}

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
  ///
  ///
  static OMPLastprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
         ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps);
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
class OMPSharedClause : public OMPVarListClause<OMPSharedClause> {
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
class OMPReductionClause : public OMPVarListClause<OMPReductionClause> {
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
        ColonLoc(ColonLoc), QualifierLoc(QualifierLoc), NameInfo(NameInfo) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPReductionClause(unsigned N)
      : OMPVarListClause<OMPReductionClause>(OMPC_reduction, SourceLocation(),
                                             SourceLocation(), SourceLocation(),
                                             N),
        ColonLoc(), QualifierLoc(), NameInfo() {}

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
  ///
  static OMPReductionClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
         NestedNameSpecifierLoc QualifierLoc,
         const DeclarationNameInfo &NameInfo, ArrayRef<Expr *> Privates,
         ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
         ArrayRef<Expr *> ReductionOps);
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
class OMPLinearClause : public OMPVarListClause<OMPLinearClause> {
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
        Modifier(Modifier), ModifierLoc(ModifierLoc), ColonLoc(ColonLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of variables.
  ///
  explicit OMPLinearClause(unsigned NumVars)
      : OMPVarListClause<OMPLinearClause>(OMPC_linear, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          NumVars),
        Modifier(OMPC_LINEAR_val), ModifierLoc(), ColonLoc() {}

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
  static OMPLinearClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
         SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
         ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep);

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
class OMPAlignedClause : public OMPVarListClause<OMPAlignedClause> {
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
class OMPCopyinClause : public OMPVarListClause<OMPCopyinClause> {
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
class OMPCopyprivateClause : public OMPVarListClause<OMPCopyprivateClause> {
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
class OMPFlushClause : public OMPVarListClause<OMPFlushClause> {
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
class OMPDependClause : public OMPVarListClause<OMPDependClause> {
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
  ///
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

} // end namespace clang

#endif // LLVM_CLANG_AST_OPENMPCLAUSE_H
