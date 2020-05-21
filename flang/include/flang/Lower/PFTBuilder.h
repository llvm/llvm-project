//===-- Lower/PFTBuilder.h -- PFT builder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Block;
}

namespace Fortran {
namespace semantics {
class SemanticsContext;
class Scope;
} // namespace semantics
namespace lower {
namespace pft {

struct Evaluation;
struct Program;
struct ModuleLikeUnit;
struct FunctionLikeUnit;

// TODO: A collection of Evaluations can obviously be any of the container
// types; leaving this as a std::list _for now_ because we reserve the right to
// insert PFT nodes in any order in O(1) time.
using EvaluationList = std::list<Evaluation>;
using LabelEvalMap = llvm::DenseMap<Fortran::parser::Label, Evaluation *>;

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
    return std::get<Ref<B>> > (u).get();
  }
  template <typename B>
  constexpr BaseType<B> *getIf() const {
    auto *ptr = std::get_if<Ref<B>>(&u);
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

/// ParentVariant is used to provide a reference to the unit a parse-tree node
/// belongs to. It is a variant of non-nullable pointers.
using ParentVariant = MutableReferenceVariant<Program, ModuleLikeUnit,
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

using OtherStmts = std::tuple<parser::FormatStmt, parser::EntryStmt,
                              parser::DataStmt, parser::NamelistStmt>;

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

using Constructs =
    std::tuple<parser::AssociateConstruct, parser::BlockConstruct,
               parser::CaseConstruct, parser::ChangeTeamConstruct,
               parser::CriticalConstruct, parser::DoConstruct,
               parser::IfConstruct, parser::SelectRankConstruct,
               parser::SelectTypeConstruct, parser::WhereConstruct,
               parser::ForallConstruct>;

using Directives =
    std::tuple<parser::CompilerDirective, parser::OpenACCConstruct,
               parser::OpenMPConstruct, parser::OmpEndLoopDirective>;

template <typename A>
static constexpr bool isActionStmt{common::HasMember<A, ActionStmts>};

template <typename A>
static constexpr bool isOtherStmt{common::HasMember<A, OtherStmts>};

template <typename A>
static constexpr bool isConstructStmt{common::HasMember<A, ConstructStmts>};

template <typename A>
static constexpr bool isConstruct{common::HasMember<A, Constructs>};

template <typename A>
static constexpr bool isDirective{common::HasMember<A, Directives>};

template <typename A>
static constexpr bool isIntermediateConstructStmt{common::HasMember<
    A, std::tuple<parser::CaseStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::SelectRankCaseStmt, parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isNopConstructStmt{common::HasMember<
    A, std::tuple<parser::EndAssociateStmt, parser::CaseStmt,
                  parser::EndSelectStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::EndIfStmt, parser::SelectRankCaseStmt,
                  parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isFunctionLike{common::HasMember<
    A, std::tuple<parser::MainProgram, parser::FunctionSubprogram,
                  parser::SubroutineSubprogram,
                  parser::SeparateModuleSubprogram>>};

using LabelSet = llvm::SmallSet<parser::Label, 5>;
using SymbolRef = common::Reference<const semantics::Symbol>;
using SymbolLabelMap = llvm::DenseMap<SymbolRef, LabelSet>;

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
    common::CombineTuples<ActionStmts, OtherStmts, ConstructStmts, Constructs,
                          Directives>;
/// Hide non-nullable pointers to the parse-tree node.
/// Build type std::variant<const A* const, const B* const, ...>
/// from EvaluationTuple type (std::tuple<A, B, ...>).
using EvaluationVariant = MakeReferenceVariant<EvaluationTuple>;

/// Function-like units contain lists of evaluations.  These can be simple
/// statements or constructs, where a construct contains its own evaluations.
struct Evaluation : EvaluationVariant {

  /// General ctor
  template <typename A>
  Evaluation(const A &a, const ParentVariant &parentVariant,
             const parser::CharBlock &position,
             const std::optional<parser::Label> &label)
      : EvaluationVariant{a},
        parentVariant{parentVariant}, position{position}, label{label} {}

  /// Construct ctor
  template <typename A>
  Evaluation(const A &a, const ParentVariant &parentVariant)
      : EvaluationVariant{a}, parentVariant{parentVariant} {
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

  /// Return the predicate:  "This is a non-initial, non-terminal construct
  /// statement."  For an IfConstruct, this is ElseIfStmt and ElseStmt.
  constexpr bool isIntermediateConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isIntermediateConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }

  /// Return the first non-nop successor of an evaluation, possibly exiting
  /// from one or more enclosing constructs.
  Evaluation &nonNopSuccessor() const {
    Evaluation *successor = lexicalSuccessor;
    if (successor && successor->isNopConstructStmt()) {
      successor = successor->parentConstruct->constructExit;
    }
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

  // FIR generation looks primarily at PFT statement (leaf) nodes.  So members
  // such as lexicalSuccessor and the various block fields are only applicable
  // to statement nodes.  One exception is that an internal construct node is
  // a convenient place for a constructExit link that applies to exits from any
  // statement within the construct.  The controlSuccessor member is used for
  // nonlexical successors, such as linking to a GOTO target.  For multiway
  // branches, controlSuccessor is set to one of the targets (might as well be
  // the first target).  Successor and exit links always target statements.
  //
  // An unstructured construct is one that contains some form of goto.  This
  // is indicated by the isUnstructured member flag, which may be set on a
  // statement and propagated to enclosing constructs.  This distinction allows
  // a structured IF or DO statement to be materialized with custom structured
  // FIR operations.  An unstructured statement is materialized as mlir
  // operation sequences that include explicit branches.
  //
  // There are two mlir::Block members.  The block member is set for statements
  // that begin a new block.  If a statement may have more than one associated
  // block, this member must be the block that would be the target of a branch
  // to the statement.  The prime example of a statement that may have multiple
  // associated blocks is NonLabelDoStmt, which may have a loop preheader block
  // for loop initialization code, and always has a header block that is the
  // target of the loop back edge.  If the NonLabelDoStmt is a concurrent loop,
  // there may be an arbitrary number of nested preheader, header, and mask
  // blocks.  Any such additional blocks in the localBlocks member are local
  // to a construct and cannot be the target of an unstructured branch.  For
  // NonLabelDoStmt, the block member designates the preheader block, which may
  // be absent if loop initialization code may be appended to a predecessor
  // block.  The primary loop header block is localBlocks[0], with additional
  // DO CONCURRENT blocks at localBlocks[1], etc.
  //
  // The printIndex member is only set for statements.  It is used for dumps
  // and does not affect FIR generation.  It may also be helpful for debugging.

  ParentVariant parentVariant;
  parser::CharBlock position{};
  std::optional<parser::Label> label{};
  std::unique_ptr<EvaluationList> evaluationList; // nested evaluations
  Evaluation *parentConstruct{nullptr};  // set for nodes below the top level
  Evaluation *lexicalSuccessor{nullptr}; // set for ActionStmt, ConstructStmt
  Evaluation *controlSuccessor{nullptr}; // set for some statements
  Evaluation *constructExit{nullptr};    // set for constructs
  bool isNewBlock{false};                // evaluation begins a new basic block
  bool isUnstructured{false};  // evaluation has unstructured control flow
  bool skip{false};            // evaluation has been processed in advance
  mlir::Block *block{nullptr}; // isNewBlock block
  llvm::SmallVector<mlir::Block *, 1> localBlocks{}; // construct local blocks
  int printIndex{0}; // (ActionStmt, ConstructStmt) evaluation index for dumps
};

using ProgramVariant =
    ReferenceVariant<parser::MainProgram, parser::FunctionSubprogram,
                     parser::SubroutineSubprogram, parser::Module,
                     parser::Submodule, parser::SeparateModuleSubprogram,
                     parser::BlockData>;
/// A program is a list of program units.
/// These units can be function like, module like, or block data.
struct ProgramUnit : ProgramVariant {
  template <typename A>
  ProgramUnit(const A &p, const ParentVariant &parentVariant)
      : ProgramVariant{p}, parentVariant{parentVariant} {}
  ProgramUnit(ProgramUnit &&) = default;
  ProgramUnit(const ProgramUnit &) = delete;

  ParentVariant parentVariant;
};

/// A variable captures an object to be created per the declaration part of a
/// function like unit.
///
/// Properties can be applied by lowering. For example, a local array that is
/// known to be very large may be transformed into a heap allocated entity by
/// lowering. That decision would be tracked in its Variable instance.
struct Variable {
  explicit Variable(const Fortran::semantics::Symbol &sym, bool global = false,
                    int depth = 0)
      : sym{&sym}, depth{depth}, global{global} {}

  const Fortran::semantics::Symbol &getSymbol() const { return *sym; }

  bool isGlobal() const { return global; }
  bool isHeapAlloc() const { return heapAlloc; }
  bool isPointer() const { return pointer; }
  bool isTarget() const { return target; }
  int getDepth() const { return depth; }

  void setHeapAlloc(bool to = true) { heapAlloc = to; }
  void setPointer(bool to = true) { pointer = to; }
  void setTarget(bool to = true) { target = to; }

private:
  const Fortran::semantics::Symbol *sym;
  int depth;
  bool global;
  bool heapAlloc{false}; // variable needs deallocation on exit
  bool pointer{false};
  bool target{false};
};

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
      const parser::MainProgram &f, const ParentVariant &parentVariant,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::FunctionSubprogram &f, const ParentVariant &parentVariant,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::SubroutineSubprogram &f, const ParentVariant &parentVariant,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(
      const parser::SeparateModuleSubprogram &f,
      const ParentVariant &parentVariant,
      const Fortran::semantics::SemanticsContext &semanticsContext);
  FunctionLikeUnit(FunctionLikeUnit &&) = default;
  FunctionLikeUnit(const FunctionLikeUnit &) = delete;

  void processSymbolTable(const Fortran::semantics::Scope &);

  std::vector<Variable> getOrderedSymbolTable() { return varList[0]; }

  bool isMainProgram() const {
    return endStmt.isA<parser::Statement<parser::EndProgramStmt>>();
  }

  /// Get the starting source location for this function like unit
  parser::CharBlock getStartingSourceLoc() {
    if (beginStmt)
      return stmtSourceLoc(*beginStmt);
    if (!evaluationList.empty())
      return evaluationList.front().position;
    return stmtSourceLoc(endStmt);
  }

  /// Returns reference to the subprogram symbol of this FunctionLikeUnit.
  /// Dies if the FunctionLikeUnit is not a subprogram.
  const semantics::Symbol &getSubprogramSymbol() const {
    assert(symbol && "not inside a procedure");
    return *symbol;
  }

  /// Helper to get location from FunctionLikeUnit begin/end statements.
  static parser::CharBlock stmtSourceLoc(const FunctionStatement &stmt) {
    return stmt.visit(common::visitors{[](const auto &x) { return x.source; }});
  }

  /// Anonymous programs do not have a begin statement
  std::optional<FunctionStatement> beginStmt;
  FunctionStatement endStmt;
  EvaluationList evaluationList;
  LabelEvalMap labelEvaluationMap;
  SymbolLabelMap assignSymbolLabelMap;
  std::list<FunctionLikeUnit> nestedFunctions;
  /// Symbol associated to this FunctionLikeUnit.
  /// Null if the FunctionLikeUnit is an anonymous program.
  /// The symbol has MainProgramDetails for named programs, otherwise it has
  /// SubprogramDetails.
  const semantics::Symbol *symbol{nullptr};
  /// Terminal basic block (if any)
  mlir::Block *finalBlock{};
  std::vector<std::vector<Variable>> varList;
};

/// Module-like units contain a list of function-like units.
struct ModuleLikeUnit : public ProgramUnit {
  // wrapper statements for module-like syntactic structures
  using ModuleStatement =
      ReferenceVariant<parser::Statement<parser::ModuleStmt>,
                       parser::Statement<parser::EndModuleStmt>,
                       parser::Statement<parser::SubmoduleStmt>,
                       parser::Statement<parser::EndSubmoduleStmt>>;

  ModuleLikeUnit(const parser::Module &m, const ParentVariant &parentVariant);
  ModuleLikeUnit(const parser::Submodule &m,
                 const ParentVariant &parentVariant);
  ~ModuleLikeUnit() = default;
  ModuleLikeUnit(ModuleLikeUnit &&) = default;
  ModuleLikeUnit(const ModuleLikeUnit &) = delete;

  ModuleStatement beginStmt;
  ModuleStatement endStmt;
  std::list<FunctionLikeUnit> nestedFunctions;
};

struct BlockDataUnit : public ProgramUnit {
  BlockDataUnit(const parser::BlockData &bd,
                const ParentVariant &parentVariant);
  BlockDataUnit(BlockDataUnit &&) = default;
  BlockDataUnit(const BlockDataUnit &) = delete;
};

/// A Program is the top-level root of the PFT.
struct Program {
  using Units = std::variant<FunctionLikeUnit, ModuleLikeUnit, BlockDataUnit>;

  Program() = default;
  Program(Program &&) = default;
  Program(const Program &) = delete;

  std::list<Units> &getUnits() { return units; }

  /// LLVM dump method on a Program.
  void dump();

private:
  std::list<Units> units;
};

} // namespace pft

/// Create a PFT (Pre-FIR Tree) from the parse tree.
///
/// A PFT is a light weight tree over the parse tree that is used to create FIR.
/// The PFT captures pointers back into the parse tree, so the parse tree must
/// not be changed between the construction of the PFT and its last use.  The
/// PFT captures a structured view of a program.  A program is a list of units.
/// A function like unit contains a list of evaluations.  An evaluation is
/// either a statement, or a construct with a nested list of evaluations.
std::unique_ptr<pft::Program>
createPFT(const parser::Program &root,
          const Fortran::semantics::SemanticsContext &semanticsContext);

/// Dumper for displaying a PFT.
void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_PFTBUILDER_H
