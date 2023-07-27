//===-- Lower/PFTBuilder.h -- PFT builder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
// PFT (Pre-FIR Tree) interface.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_PFTBUILDER_H
#define FORTRAN_LOWER_PFTBUILDER_H

#include "flang/Common/reference.h"
#include "flang/Common/template.h"
#include "flang/Lower/HostAssociations.h"
#include "flang/Lower/PFTDefs.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::lower::pft {

struct Evaluation;
struct Program;
struct ModuleLikeUnit;
struct FunctionLikeUnit;

using EvaluationList = std::list<Evaluation>;

/// Provide a variant like container that can hold references. It can hold
/// constant or mutable references. It is used in the other classes to provide
/// union of const references to parse-tree nodes.
template <bool isConst, typename... A>
class ReferenceVariantBase {
public:
  template <typename B>
  using BaseType = std::conditional_t<isConst, const B, B>;
  template <typename B>
  using Ref = common::Reference<BaseType<B>>;

  ReferenceVariantBase() = delete;
  ReferenceVariantBase(std::variant<Ref<A>...> b) : u(b) {}
  template <typename T>
  ReferenceVariantBase(Ref<T> b) : u(b) {}

  template <typename B>
  constexpr BaseType<B> &get() const {
    return std::get<Ref<B>>(u).get();
  }
  template <typename B>
  constexpr BaseType<B> &getStatement() const {
    return std::get<Ref<parser::Statement<B>>>(u).get().statement;
  }
  template <typename B>
  constexpr BaseType<B> *getIf() const {
    const Ref<B> *ptr = std::get_if<Ref<B>>(&u);
    return ptr ? &ptr->get() : nullptr;
  }
  template <typename B>
  constexpr bool isA() const {
    return std::holds_alternative<Ref<B>>(u);
  }
  template <typename VISITOR>
  constexpr auto visit(VISITOR &&visitor) const {
    return std::visit(
        common::visitors{[&visitor](auto ref) { return visitor(ref.get()); }},
        u);
  }

private:
  std::variant<Ref<A>...> u;
};
template <typename... A>
using ReferenceVariant = ReferenceVariantBase<true, A...>;
template <typename... A>
using MutableReferenceVariant = ReferenceVariantBase<false, A...>;

/// PftNode is used to provide a reference to the unit a parse-tree node
/// belongs to. It is a variant of non-nullable pointers.
using PftNode = MutableReferenceVariant<Program, ModuleLikeUnit,
                                        FunctionLikeUnit, Evaluation>;

/// Classify the parse-tree nodes from ExecutablePartConstruct

using ActionStmts = std::tuple<
    parser::AllocateStmt, parser::AssignmentStmt, parser::BackspaceStmt,
    parser::CallStmt, parser::CloseStmt, parser::ContinueStmt,
    parser::CycleStmt, parser::DeallocateStmt, parser::EndfileStmt,
    parser::EventPostStmt, parser::EventWaitStmt, parser::ExitStmt,
    parser::FailImageStmt, parser::FlushStmt, parser::FormTeamStmt,
    parser::GotoStmt, parser::IfStmt, parser::InquireStmt, parser::LockStmt,
    parser::NullifyStmt, parser::OpenStmt, parser::PointerAssignmentStmt,
    parser::PrintStmt, parser::ReadStmt, parser::ReturnStmt, parser::RewindStmt,
    parser::StopStmt, parser::SyncAllStmt, parser::SyncImagesStmt,
    parser::SyncMemoryStmt, parser::SyncTeamStmt, parser::UnlockStmt,
    parser::WaitStmt, parser::WhereStmt, parser::WriteStmt,
    parser::ComputedGotoStmt, parser::ForallStmt, parser::ArithmeticIfStmt,
    parser::AssignStmt, parser::AssignedGotoStmt, parser::PauseStmt>;

using OtherStmts = std::tuple<parser::EntryStmt, parser::FormatStmt>;

using ConstructStmts = std::tuple<
    parser::AssociateStmt, parser::EndAssociateStmt, parser::BlockStmt,
    parser::EndBlockStmt, parser::SelectCaseStmt, parser::CaseStmt,
    parser::EndSelectStmt, parser::ChangeTeamStmt, parser::EndChangeTeamStmt,
    parser::CriticalStmt, parser::EndCriticalStmt, parser::NonLabelDoStmt,
    parser::EndDoStmt, parser::IfThenStmt, parser::ElseIfStmt, parser::ElseStmt,
    parser::EndIfStmt, parser::SelectRankStmt, parser::SelectRankCaseStmt,
    parser::SelectTypeStmt, parser::TypeGuardStmt, parser::WhereConstructStmt,
    parser::MaskedElsewhereStmt, parser::ElsewhereStmt, parser::EndWhereStmt,
    parser::ForallConstructStmt, parser::EndForallStmt>;

using EndStmts =
    std::tuple<parser::EndProgramStmt, parser::EndFunctionStmt,
               parser::EndSubroutineStmt, parser::EndMpSubprogramStmt>;

using Constructs =
    std::tuple<parser::AssociateConstruct, parser::BlockConstruct,
               parser::CaseConstruct, parser::ChangeTeamConstruct,
               parser::CriticalConstruct, parser::DoConstruct,
               parser::IfConstruct, parser::SelectRankConstruct,
               parser::SelectTypeConstruct, parser::WhereConstruct,
               parser::ForallConstruct>;

using Directives =
    std::tuple<parser::CompilerDirective, parser::OpenACCConstruct,
               parser::OpenACCDeclarativeConstruct, parser::OpenMPConstruct,
               parser::OpenMPDeclarativeConstruct, parser::OmpEndLoopDirective>;

using DeclConstructs = std::tuple<parser::OpenMPDeclarativeConstruct,
                                  parser::OpenACCDeclarativeConstruct>;

template <typename A>
static constexpr bool isActionStmt{common::HasMember<A, ActionStmts>};

template <typename A>
static constexpr bool isOtherStmt{common::HasMember<A, OtherStmts>};

template <typename A>
static constexpr bool isConstructStmt{common::HasMember<A, ConstructStmts>};

template <typename A>
static constexpr bool isEndStmt{common::HasMember<A, EndStmts>};

template <typename A>
static constexpr bool isConstruct{common::HasMember<A, Constructs>};

template <typename A>
static constexpr bool isDirective{common::HasMember<A, Directives>};

template <typename A>
static constexpr bool isDeclConstruct{common::HasMember<A, DeclConstructs>};

template <typename A>
static constexpr bool isIntermediateConstructStmt{common::HasMember<
    A, std::tuple<parser::CaseStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::SelectRankCaseStmt, parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isNopConstructStmt{common::HasMember<
    A, std::tuple<parser::CaseStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::EndIfStmt, parser::SelectRankCaseStmt,
                  parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isExecutableDirective{common::HasMember<
    A, std::tuple<parser::CompilerDirective, parser::OpenACCConstruct,
                  parser::OpenMPConstruct>>};

template <typename A>
static constexpr bool isFunctionLike{common::HasMember<
    A, std::tuple<parser::MainProgram, parser::FunctionSubprogram,
                  parser::SubroutineSubprogram,
                  parser::SeparateModuleSubprogram>>};

template <typename A>
struct MakeReferenceVariantHelper {};
template <typename... A>
struct MakeReferenceVariantHelper<std::variant<A...>> {
  using type = ReferenceVariant<A...>;
};
template <typename... A>
struct MakeReferenceVariantHelper<std::tuple<A...>> {
  using type = ReferenceVariant<A...>;
};
template <typename A>
using MakeReferenceVariant = typename MakeReferenceVariantHelper<A>::type;

using EvaluationTuple =
    common::CombineTuples<ActionStmts, OtherStmts, ConstructStmts, EndStmts,
                          Constructs, Directives>;
/// Hide non-nullable pointers to the parse-tree node.
/// Build type std::variant<const A* const, const B* const, ...>
/// from EvaluationTuple type (std::tuple<A, B, ...>).
using EvaluationVariant = MakeReferenceVariant<EvaluationTuple>;

/// Function-like units contain lists of evaluations. These can be simple
/// statements or constructs, where a construct contains its own evaluations.
struct Evaluation : EvaluationVariant {

  /// General ctor
  template <typename A>
  Evaluation(const A &a, const PftNode &parent,
             const parser::CharBlock &position,
             const std::optional<parser::Label> &label)
      : EvaluationVariant{a}, parent{parent}, position{position}, label{label} {
  }

  /// Construct and Directive ctor
  template <typename A>
  Evaluation(const A &a, const PftNode &parent)
      : EvaluationVariant{a}, parent{parent} {
    static_assert(pft::isConstruct<A> || pft::isDirective<A>,
                  "must be a construct or directive");
  }

  /// Evaluation classification predicates.
  constexpr bool isActionStmt() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isActionStmt<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isOtherStmt() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isOtherStmt<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }
  constexpr bool isEndStmt() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isEndStmt<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isConstruct() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isConstruct<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isDirective() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isDirective<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isNopConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isNopConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }
  constexpr bool isExecutableDirective() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isExecutableDirective<std::decay_t<decltype(r)>>;
    }});
  }

  /// Return the predicate:  "This is a non-initial, non-terminal construct
  /// statement."  For an IfConstruct, this is ElseIfStmt and ElseStmt.
  constexpr bool isIntermediateConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isIntermediateConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }

  LLVM_DUMP_METHOD void dump() const;

  /// Return the first non-nop successor of an evaluation, possibly exiting
  /// from one or more enclosing constructs.
  Evaluation &nonNopSuccessor() const {
    Evaluation *successor = lexicalSuccessor;
    if (successor && successor->isNopConstructStmt())
      successor = successor->parentConstruct->constructExit;
    assert(successor && "missing successor");
    return *successor;
  }

  /// Return true if this Evaluation has at least one nested evaluation.
  bool hasNestedEvaluations() const {
    return evaluationList && !evaluationList->empty();
  }

  /// Return nested evaluation list.
  EvaluationList &getNestedEvaluations() {
    assert(evaluationList && "no nested evaluations");
    return *evaluationList;
  }

  Evaluation &getFirstNestedEvaluation() {
    assert(hasNestedEvaluations() && "no nested evaluations");
    return evaluationList->front();
  }

  Evaluation &getLastNestedEvaluation() {
    assert(hasNestedEvaluations() && "no nested evaluations");
    return evaluationList->back();
  }

  /// Return the FunctionLikeUnit containing this evaluation (or nullptr).
  FunctionLikeUnit *getOwningProcedure() const;

  bool lowerAsStructured() const;
  bool lowerAsUnstructured() const;
  bool forceAsUnstructured() const;

  // FIR generation looks primarily at PFT ActionStmt and ConstructStmt leaf
  // nodes. Members such as lexicalSuccessor and block are applicable only
  // to these nodes, plus some directives. The controlSuccessor member is
  // used for nonlexical successors, such as linking to a GOTO target. For
  // multiway branches, it is set to the first target. Successor and exit
  // links always target statements or directives. An internal Construct
  // node has a constructExit link that applies to exits from anywhere within
  // the construct.
  //
  // An unstructured construct is one that contains some form of goto. This
  // is indicated by the isUnstructured member flag, which may be set on a
  // statement and propagated to enclosing constructs. This distinction allows
  // a structured IF or DO statement to be materialized with custom structured
  // FIR operations. An unstructured statement is materialized as mlir
  // operation sequences that include explicit branches.
  //
  // The block member is set for statements that begin a new block. This
  // block is the target of any branch to the statement. Statements may have
  // additional (unstructured) "local" blocks, but such blocks cannot be the
  // target of any explicit branch. The primary example of an (unstructured)
  // statement that may have multiple associated blocks is NonLabelDoStmt,
  // which may have a loop preheader block for loop initialization code (the
  // block member), and always has a "local" header block that is the target
  // of the loop back edge. If the NonLabelDoStmt is a concurrent loop, it
  // may be associated with an arbitrary number of nested preheader, header,
  // and mask blocks.
  //
  // The printIndex member is only set for statements. It is used for dumps
  // (and debugging) and does not affect FIR generation.

  PftNode parent;
  parser::CharBlock position{};
  std::optional<parser::Label> label{};
  std::unique_ptr<EvaluationList> evaluationList; // nested evaluations
  Evaluation *parentConstruct{nullptr};  // set for nodes below the top level
  Evaluation *lexicalSuccessor{nullptr}; // set for leaf nodes, some directives
  Evaluation *controlSuccessor{nullptr}; // set for some leaf nodes
  Evaluation *constructExit{nullptr};    // set for constructs
  bool isNewBlock{false};                // evaluation begins a new basic block
  bool isUnstructured{false};  // evaluation has unstructured control flow
  bool negateCondition{false}; // If[Then]Stmt condition must be negated
  bool activeConstruct{false}; // temporarily set for some constructs
  mlir::Block *block{nullptr}; // isNewBlock block (ActionStmt, ConstructStmt)
  int printIndex{0}; // (ActionStmt, ConstructStmt) evaluation index for dumps
};

using ProgramVariant =
    ReferenceVariant<parser::MainProgram, parser::FunctionSubprogram,
                     parser::SubroutineSubprogram, parser::Module,
                     parser::Submodule, parser::SeparateModuleSubprogram,
                     parser::BlockData, parser::CompilerDirective>;
/// A program is a list of program units.
/// These units can be function like, module like, or block data.
struct ProgramUnit : ProgramVariant {
  template <typename A>
  ProgramUnit(const A &p, const PftNode &parent)
      : ProgramVariant{p}, parent{parent} {}
  ProgramUnit(ProgramUnit &&) = default;
  ProgramUnit(const ProgramUnit &) = delete;

  PftNode parent;
};

/// A variable captures an object to be created per the declaration part of a
/// function like unit.
///
/// Fortran EQUIVALENCE statements are a mechanism that introduces aliasing
/// between named variables. The set of overlapping aliases will materialize a
/// generic store object with a designated offset and size. Participant
/// symbols will simply be pointers into the aggregate store.
///
/// EQUIVALENCE can also interact with COMMON and other global variables to
/// imply aliasing between (subparts of) a global and other local variable
/// names.
///
/// Properties can be applied by lowering. For example, a local array that is
/// known to be very large may be transformed into a heap allocated entity by
/// lowering. That decision would be tracked in its Variable instance.
struct Variable {
  /// Most variables are nominal and require the allocation of local/global
  /// storage space. A nominal variable may also be an alias for some other
  /// (subpart) of storage.
  struct Nominal {
    Nominal(const semantics::Symbol *symbol, int depth, bool global)
        : symbol{symbol}, depth{depth}, global{global} {}
    const semantics::Symbol *symbol{};

    bool isGlobal() const { return global; }

    int depth{};
    bool global{};
    bool heapAlloc{}; // variable needs deallocation on exit
    bool pointer{};
    bool target{};
    bool aliaser{}; // participates in EQUIVALENCE union
    std::size_t aliasOffset{};
  };

  /// <offset, size> pair
  using Interval = std::tuple<std::size_t, std::size_t>;

  /// An interval of storage is a contiguous block of memory to be allocated or
  /// mapped onto another variable. Aliasing variables will be pointers into
  /// interval stores and may overlap each other.
  struct AggregateStore {
    AggregateStore(Interval &&interval,
                   const Fortran::semantics::Symbol &namingSym,
                   bool isGlobal = false)
        : interval{std::move(interval)}, namingSymbol{&namingSym},
          isGlobalAggregate{isGlobal} {}
    AggregateStore(const semantics::Symbol &initialValueSym,
                   const semantics::Symbol &namingSym, bool isGlobal = false)
        : interval{initialValueSym.offset(), initialValueSym.size()},
          namingSymbol{&namingSym}, initialValueSymbol{&initialValueSym},
          isGlobalAggregate{isGlobal} {};

    bool isGlobal() const { return isGlobalAggregate; }
    /// Get offset of the aggregate inside its scope.
    std::size_t getOffset() const { return std::get<0>(interval); }
    /// Returns symbols holding the aggregate initial value if any.
    const semantics::Symbol *getInitialValueSymbol() const {
      return initialValueSymbol;
    }
    /// Returns the symbol that gives its name to the aggregate.
    const semantics::Symbol &getNamingSymbol() const { return *namingSymbol; }
    /// Scope to which the aggregates belongs to.
    const semantics::Scope &getOwningScope() const {
      return getNamingSymbol().owner();
    }
    /// <offset, size> of the aggregate in its scope.
    Interval interval{};
    /// Symbol that gives its name to the aggregate. Always set by constructor.
    const semantics::Symbol *namingSymbol;
    /// Compiler generated symbol with the aggregate initial value if any.
    const semantics::Symbol *initialValueSymbol = nullptr;
    /// Is this a global aggregate?
    bool isGlobalAggregate;
  };

  explicit Variable(const Fortran::semantics::Symbol &sym, bool global = false,
                    int depth = 0)
      : var{Nominal(&sym, depth, global)} {}
  explicit Variable(AggregateStore &&istore) : var{std::move(istore)} {}

  /// Return the front-end symbol for a nominal variable.
  const Fortran::semantics::Symbol &getSymbol() const {
    assert(hasSymbol() && "variable is not nominal");
    return *std::get<Nominal>(var).symbol;
  }

  /// Return the aggregate store.
  const AggregateStore &getAggregateStore() const {
    assert(isAggregateStore());
    return std::get<AggregateStore>(var);
  }

  /// Return the interval range of an aggregate store.
  const Interval &getInterval() const {
    assert(isAggregateStore());
    return std::get<AggregateStore>(var).interval;
  }

  /// Only nominal variable have front-end symbols.
  bool hasSymbol() const { return std::holds_alternative<Nominal>(var); }

  /// Is this an aggregate store?
  bool isAggregateStore() const {
    return std::holds_alternative<AggregateStore>(var);
  }

  /// Is this variable a global?
  bool isGlobal() const {
    return std::visit([](const auto &x) { return x.isGlobal(); }, var);
  }

  /// Is this a module or submodule variable?
  bool isModuleOrSubmoduleVariable() const {
    const semantics::Scope *scope = getOwningScope();
    return scope && scope->kind() == Fortran::semantics::Scope::Kind::Module;
  }

  const Fortran::semantics::Scope *getOwningScope() const {
    return std::visit(
        common::visitors{
            [](const Nominal &x) { return &x.symbol->GetUltimate().owner(); },
            [](const AggregateStore &agg) { return &agg.getOwningScope(); }},
        var);
  }

  bool isHeapAlloc() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->heapAlloc;
    return false;
  }
  bool isPointer() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->pointer;
    return false;
  }
  bool isTarget() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->target;
    return false;
  }

  /// An alias(er) is a variable that is part of a EQUIVALENCE that is allocated
  /// locally on the stack.
  bool isAlias() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->aliaser;
    return false;
  }
  std::size_t getAliasOffset() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->aliasOffset;
    return 0;
  }
  void setAlias(std::size_t offset) {
    if (auto *s = std::get_if<Nominal>(&var)) {
      s->aliaser = true;
      s->aliasOffset = offset;
    } else {
      llvm_unreachable("not a nominal var");
    }
  }

  void setHeapAlloc(bool to = true) {
    if (auto *s = std::get_if<Nominal>(&var))
      s->heapAlloc = to;
    else
      llvm_unreachable("not a nominal var");
  }
  void setPointer(bool to = true) {
    if (auto *s = std::get_if<Nominal>(&var))
      s->pointer = to;
    else
      llvm_unreachable("not a nominal var");
  }
  void setTarget(bool to = true) {
    if (auto *s = std::get_if<Nominal>(&var))
      s->target = to;
    else
      llvm_unreachable("not a nominal var");
  }

  /// The depth is recorded for nominal variables as a debugging aid.
  int getDepth() const {
    if (auto *s = std::get_if<Nominal>(&var))
      return s->depth;
    return 0;
  }

  LLVM_DUMP_METHOD void dump() const;

private:
  std::variant<Nominal, AggregateStore> var;
};

using VariableList = std::vector<Variable>;
using ScopeVariableListMap =
    std::map<const Fortran::semantics::Scope *, VariableList>;

/// Find or create an ordered list of the equivalence sets and variables that
/// appear in \p scope. The result is cached in \p map.
const VariableList &getScopeVariableList(const Fortran::semantics::Scope &scope,
                                         ScopeVariableListMap &map);

/// Create an ordered list of the equivalence sets and variables that appear in
/// \p scope. The result is not cached.
VariableList getScopeVariableList(const Fortran::semantics::Scope &scope);

/// Create an ordered list of the equivalence sets and variables that \p symbol
/// depends on. \p symbol itself will be the last variable in the list.
VariableList getDependentVariableList(const Fortran::semantics::Symbol &);

void dump(VariableList &, std::string s = {}); // `s` is an optional dump label

/// Function-like units may contain evaluations (executable statements) and
/// nested function-like units (internal procedures and function statements).
struct FunctionLikeUnit : public ProgramUnit {
  // wrapper statements for function-like syntactic structures
  using FunctionStatement =
      ReferenceVariant<parser::Statement<parser::ProgramStmt>,
                       parser::Statement<parser::EndProgramStmt>,
                       parser::Statement<parser::FunctionStmt>,
                       parser::Statement<parser::EndFunctionStmt>,
                       parser::Statement<parser::SubroutineStmt>,
                       parser::Statement<parser::EndSubroutineStmt>,
                       parser::Statement<parser::MpSubprogramStmt>,
                       parser::Statement<parser::EndMpSubprogramStmt>>;

  FunctionLikeUnit(
      const parser::MainProgram &f, const PftNode &parent,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::FunctionSubprogram &f, const PftNode &parent,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::SubroutineSubprogram &f, const PftNode &parent,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::SeparateModuleSubprogram &f, const PftNode &parent,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(FunctionLikeUnit &&) = default;
  FunctionLikeUnit(const FunctionLikeUnit &) = delete;

  bool isMainProgram() const {
    return endStmt.isA<parser::Statement<parser::EndProgramStmt>>();
  }

  /// Get the starting source location for this function like unit
  parser::CharBlock getStartingSourceLoc() const;

  void setActiveEntry(int entryIndex) {
    assert(entryIndex >= 0 && entryIndex < (int)entryPointList.size() &&
           "invalid entry point index");
    activeEntry = entryIndex;
  }

  /// Return a reference to the subprogram symbol of this FunctionLikeUnit.
  /// This should not be called if the FunctionLikeUnit is the main program
  /// since anonymous main programs do not have a symbol.
  const semantics::Symbol &getSubprogramSymbol() const {
    const semantics::Symbol *symbol = entryPointList[activeEntry].first;
    if (!symbol)
      llvm::report_fatal_error(
          "not inside a procedure; do not call on main program.");
    return *symbol;
  }

  /// Return a pointer to the main program symbol for named programs
  /// Return the null pointer for anonymous programs
  const semantics::Symbol *getMainProgramSymbol() const {
    if (!isMainProgram()) {
      llvm::report_fatal_error("call only on main program.");
    }
    return entryPointList[activeEntry].first;
  }

  /// Return a pointer to the current entry point Evaluation.
  /// This is null for a primary entry point.
  Evaluation *getEntryEval() const {
    return entryPointList[activeEntry].second;
  }

  //===--------------------------------------------------------------------===//
  // Host associations
  //===--------------------------------------------------------------------===//

  void setHostAssociatedSymbols(
      const llvm::SetVector<const semantics::Symbol *> &symbols) {
    hostAssociations.addSymbolsToBind(symbols, getScope());
  }

  /// Return the host associations, if any, from the parent (host) procedure.
  /// Crashes if the parent is not a procedure.
  HostAssociations &parentHostAssoc();

  /// Return true iff the parent is a procedure and the parent has a non-empty
  /// set of host associations that are conveyed through an extra tuple
  /// argument.
  bool parentHasTupleHostAssoc();

  /// Return true iff the parent is a procedure and the parent has a non-empty
  /// set of host associations for variables.
  bool parentHasHostAssoc();

  /// Return the host associations for this function like unit. The list of host
  /// associations are kept in the host procedure.
  HostAssociations &getHostAssoc() { return hostAssociations; }

  LLVM_DUMP_METHOD void dump() const;

  /// Get the function scope.
  const Fortran::semantics::Scope &getScope() const { return *scope; }

  /// Anonymous programs do not have a begin statement.
  std::optional<FunctionStatement> beginStmt;
  FunctionStatement endStmt;
  const semantics::Scope *scope;
  EvaluationList evaluationList;
  LabelEvalMap labelEvaluationMap;
  SymbolLabelMap assignSymbolLabelMap;
  std::list<FunctionLikeUnit> nestedFunctions;
  /// <Symbol, Evaluation> pairs for each entry point. The pair at index 0
  /// is the primary entry point; remaining pairs are alternate entry points.
  /// The primary entry point symbol is Null for an anonymous program.
  /// A named program symbol has MainProgramDetails. Other symbols have
  /// SubprogramDetails. Evaluations are filled in for alternate entries.
  llvm::SmallVector<std::pair<const semantics::Symbol *, Evaluation *>, 1>
      entryPointList{std::pair{nullptr, nullptr}};
  /// Current index into entryPointList. Index 0 is the primary entry point.
  int activeEntry = 0;
  /// Primary result for function subprograms with alternate entries. This
  /// is one of the largest result values, not necessarily the first one.
  const semantics::Symbol *primaryResult{nullptr};
  bool mayModifyRoundingMode{false};
  /// Terminal basic block (if any)
  mlir::Block *finalBlock{};
  HostAssociations hostAssociations;
};

/// Module-like units contain a list of function-like units.
struct ModuleLikeUnit : public ProgramUnit {
  // wrapper statements for module-like syntactic structures
  using ModuleStatement =
      ReferenceVariant<parser::Statement<parser::ModuleStmt>,
                       parser::Statement<parser::EndModuleStmt>,
                       parser::Statement<parser::SubmoduleStmt>,
                       parser::Statement<parser::EndSubmoduleStmt>>;

  ModuleLikeUnit(const parser::Module &m, const PftNode &parent);
  ModuleLikeUnit(const parser::Submodule &m, const PftNode &parent);
  ~ModuleLikeUnit() = default;
  ModuleLikeUnit(ModuleLikeUnit &&) = default;
  ModuleLikeUnit(const ModuleLikeUnit &) = delete;

  LLVM_DUMP_METHOD void dump() const;

  /// Get the starting source location for this module like unit.
  parser::CharBlock getStartingSourceLoc() const;

  /// Get the module scope.
  const Fortran::semantics::Scope &getScope() const;

  ModuleStatement beginStmt;
  ModuleStatement endStmt;
  std::list<FunctionLikeUnit> nestedFunctions;
  EvaluationList evaluationList;
};

/// Block data units contain the variables and data initializers for common
/// blocks, etc.
struct BlockDataUnit : public ProgramUnit {
  BlockDataUnit(const parser::BlockData &bd, const PftNode &parent,
                const Fortran::semantics::SemanticsContext &semanticsContext);
  BlockDataUnit(BlockDataUnit &&) = default;
  BlockDataUnit(const BlockDataUnit &) = delete;

  LLVM_DUMP_METHOD void dump() const;

  const Fortran::semantics::Scope &symTab; // symbol table
};

// Top level compiler directives
struct CompilerDirectiveUnit : public ProgramUnit {
  CompilerDirectiveUnit(const parser::CompilerDirective &directive,
                        const PftNode &parent)
      : ProgramUnit{directive, parent} {};
  CompilerDirectiveUnit(CompilerDirectiveUnit &&) = default;
  CompilerDirectiveUnit(const CompilerDirectiveUnit &) = delete;
};

/// A Program is the top-level root of the PFT.
struct Program {
  using Units = std::variant<FunctionLikeUnit, ModuleLikeUnit, BlockDataUnit,
                             CompilerDirectiveUnit>;

  Program(semantics::CommonBlockList &&commonBlocks)
      : commonBlocks{std::move(commonBlocks)} {}
  Program(Program &&) = default;
  Program(const Program &) = delete;

  const std::list<Units> &getUnits() const { return units; }
  std::list<Units> &getUnits() { return units; }
  const semantics::CommonBlockList &getCommonBlocks() const {
    return commonBlocks;
  }
  ScopeVariableListMap &getScopeVariableListMap() {
    return scopeVariableListMap;
  }

  /// LLVM dump method on a Program.
  LLVM_DUMP_METHOD void dump() const;

private:
  std::list<Units> units;
  semantics::CommonBlockList commonBlocks;
  ScopeVariableListMap scopeVariableListMap; // module and submodule scopes
};

/// Helper to get location from FunctionLikeUnit/ModuleLikeUnit begin/end
/// statements.
template <typename T>
static parser::CharBlock stmtSourceLoc(const T &stmt) {
  return stmt.visit(common::visitors{[](const auto &x) { return x.source; }});
}

/// Get the first PFT ancestor node that has type ParentType.
template <typename ParentType, typename A>
ParentType *getAncestor(A &node) {
  if (auto *seekedParent = node.parent.template getIf<ParentType>())
    return seekedParent;
  return node.parent.visit(common::visitors{
      [](Program &p) -> ParentType * { return nullptr; },
      [](auto &p) -> ParentType * { return getAncestor<ParentType>(p); }});
}

/// Get the "global" scopeVariableListMap, stored in the pft root node.
template <typename A>
ScopeVariableListMap &getScopeVariableListMap(A &node) {
  Program *pftRoot = getAncestor<Program>(node);
  assert(pftRoot && "pft must have a root");
  return pftRoot->getScopeVariableListMap();
}

/// Call the provided \p callBack on all symbols that are referenced inside \p
/// funit.
void visitAllSymbols(const FunctionLikeUnit &funit,
                     std::function<void(const semantics::Symbol &)> callBack);

/// Call the provided \p callBack on all symbols that are referenced inside \p
/// eval region.
void visitAllSymbols(const Evaluation &eval,
                     std::function<void(const semantics::Symbol &)> callBack);

} // namespace Fortran::lower::pft

namespace Fortran::lower {
/// Create a PFT (Pre-FIR Tree) from the parse tree.
///
/// A PFT is a light weight tree over the parse tree that is used to create FIR.
/// The PFT captures pointers back into the parse tree, so the parse tree must
/// not be changed between the construction of the PFT and its last use. The
/// PFT captures a structured view of a program. A program is a list of units.
/// A function like unit contains a list of evaluations. An evaluation is
/// either a statement, or a construct with a nested list of evaluations.
std::unique_ptr<pft::Program>
createPFT(const parser::Program &root,
          const Fortran::semantics::SemanticsContext &semanticsContext);

/// Dumper for displaying a PFT.
void dumpPFT(llvm::raw_ostream &outputStream, const pft::Program &pft);
} // namespace Fortran::lower

#endif // FORTRAN_LOWER_PFTBUILDER_H
