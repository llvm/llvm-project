//===-- PFTBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/IntervalSet.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-pft"

static llvm::cl::opt<bool> clDisableStructuredFir(
    "no-structured-fir", llvm::cl::desc("disable generation of structured FIR"),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<bool> nonRecursiveProcedures(
    "non-recursive-procedures",
    llvm::cl::desc("Make procedures non-recursive by default. This was the "
                   "default for all Fortran standards prior to 2018."),
    llvm::cl::init(/*2018 standard=*/false));

static llvm::cl::opt<bool> mainProgramGlobals(
    "main-program-globals",
    llvm::cl::desc(
        "Allocate all variables in the main program as global variables and "
        "not on the stack regardless of type, kind, and rank."),
    llvm::cl::init(/*2018 standard=*/false), llvm::cl::Hidden);

using namespace Fortran;

namespace {
/// Helpers to unveil parser node inside Fortran::parser::Statement<>,
/// Fortran::parser::UnlabeledStatement, and Fortran::common::Indirection<>
template <typename A>
struct RemoveIndirectionHelper {
  using Type = A;
};
template <typename A>
struct RemoveIndirectionHelper<common::Indirection<A>> {
  using Type = A;
};

template <typename A>
struct UnwrapStmt {
  static constexpr bool isStmt{false};
};
template <typename A>
struct UnwrapStmt<parser::Statement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::Statement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source},
        label{a.label} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};
template <typename A>
struct UnwrapStmt<parser::UnlabeledStatement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::UnlabeledStatement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time.  So one goal here is to
/// limit the bridge to one such instantiation.
class PFTBuilder {
public:
  PFTBuilder(const semantics::SemanticsContext &semanticsContext)
      : pgm{std::make_unique<lower::pft::Program>(
            semanticsContext.GetCommonBlocks())},
        semanticsContext{semanticsContext} {
    lower::pft::PftNode pftRoot{*pgm.get()};
    pftParentStack.push_back(pftRoot);
  }

  /// Get the result
  std::unique_ptr<lower::pft::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    if constexpr (lower::pft::isFunctionLike<A>) {
      return enterFunction(a, semanticsContext);
    } else if constexpr (lower::pft::isConstruct<A> ||
                         lower::pft::isDirective<A>) {
      return enterConstructOrDirective(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (lower::pft::isConstructStmt<T> ||
                    lower::pft::isOtherStmt<T>) {
        addEvaluation(lower::pft::Evaluation{
            stmt.unwrapped, pftParentStack.back(), stmt.position, stmt.label});
        return false;
      } else if constexpr (std::is_same_v<T, parser::ActionStmt>) {
        return std::visit(
            common::visitors{
                [&](const common::Indirection<parser::IfStmt> &x) {
                  convertIfStmt(x.value(), stmt.position, stmt.label);
                  return false;
                },
                [&](const auto &x) {
                  addEvaluation(lower::pft::Evaluation{
                      removeIndirection(x), pftParentStack.back(),
                      stmt.position, stmt.label});
                  return true;
                },
            },
            stmt.unwrapped.u);
      }
    }
    return true;
  }

  /// Convert an IfStmt into an IfConstruct, retaining the IfStmt as the
  /// first statement of the construct.
  void convertIfStmt(const parser::IfStmt &ifStmt, parser::CharBlock position,
                     std::optional<parser::Label> label) {
    // Generate a skeleton IfConstruct parse node.  Its components are never
    // referenced.  The actual components are available via the IfConstruct
    // evaluation's nested evaluationList, with the ifStmt in the position of
    // the otherwise normal IfThenStmt.  Caution: All other PFT nodes reference
    // front end generated parse nodes; this is an exceptional case.
    static const auto ifConstruct = parser::IfConstruct{
        parser::Statement<parser::IfThenStmt>{
            std::nullopt,
            parser::IfThenStmt{
                std::optional<parser::Name>{},
                parser::ScalarLogicalExpr{parser::LogicalExpr{parser::Expr{
                    parser::LiteralConstant{parser::LogicalLiteralConstant{
                        false, std::optional<parser::KindParam>{}}}}}}}},
        parser::Block{}, std::list<parser::IfConstruct::ElseIfBlock>{},
        std::optional<parser::IfConstruct::ElseBlock>{},
        parser::Statement<parser::EndIfStmt>{std::nullopt,
                                             parser::EndIfStmt{std::nullopt}}};
    enterConstructOrDirective(ifConstruct);
    addEvaluation(
        lower::pft::Evaluation{ifStmt, pftParentStack.back(), position, label});
    Pre(std::get<parser::UnlabeledStatement<parser::ActionStmt>>(ifStmt.t));
    static const auto endIfStmt = parser::EndIfStmt{std::nullopt};
    addEvaluation(
        lower::pft::Evaluation{endIfStmt, pftParentStack.back(), {}, {}});
    exitConstructOrDirective();
  }

  template <typename A>
  constexpr void Post(const A &) {
    if constexpr (lower::pft::isFunctionLike<A>) {
      exitFunction();
    } else if constexpr (lower::pft::isConstruct<A> ||
                         lower::pft::isDirective<A>) {
      exitConstructOrDirective();
    }
  }

  // Module like
  bool Pre(const parser::Module &node) { return enterModule(node); }
  bool Pre(const parser::Submodule &node) { return enterModule(node); }

  void Post(const parser::Module &) { exitModule(); }
  void Post(const parser::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const parser::BlockData &node) {
    addUnit(lower::pft::BlockDataUnit{node, pftParentStack.back(),
                                      semanticsContext});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const parser::Statement<parser::ForallAssignmentStmt> &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return lower::pft::Evaluation{x, pftParentStack.back(),
                                        statement.source, statement.label};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::WhereBodyConstruct &whereBody) {
    return std::visit(
        common::visitors{
            [&](const parser::Statement<parser::AssignmentStmt> &stmt) {
              // Not caught as other AssignmentStmt because it is not
              // wrapped in a parser::ActionStmt.
              addEvaluation(lower::pft::Evaluation{stmt.statement,
                                                   pftParentStack.back(),
                                                   stmt.source, stmt.label});
              return false;
            },
            [&](const auto &) { return true; },
        },
        whereBody.u);
  }

  // CompilerDirective have special handling in case they are top level
  // directives (i.e. they do not belong to a ProgramUnit).
  bool Pre(const parser::CompilerDirective &directive) {
    assert(pftParentStack.size() > 0 &&
           "At least the Program must be a parent");
    if (pftParentStack.back().isA<lower::pft::Program>()) {
      addUnit(
          lower::pft::CompilerDirectiveUnit(directive, pftParentStack.back()));
      return false;
    }
    return enterConstructOrDirective(directive);
  }

private:
  /// Initialize a new module-like unit and make it the builder's focus.
  template <typename A>
  bool enterModule(const A &func) {
    Fortran::lower::pft::ModuleLikeUnit &unit =
        addUnit(lower::pft::ModuleLikeUnit{func, pftParentStack.back()});
    functionList = &unit.nestedFunctions;
    pushEvaluationList(&unit.evaluationList);
    pftParentStack.emplace_back(unit);
    return true;
  }

  void exitModule() {
    if (!evaluationListStack.empty())
      popEvaluationList();
    pftParentStack.pop_back();
    resetFunctionState();
  }

  /// Add the end statement Evaluation of a sub/program to the PFT.
  /// There may be intervening internal subprogram definitions between
  /// prior statements and this end statement.
  void endFunctionBody() {
    if (evaluationListStack.empty())
      return;
    auto evaluationList = evaluationListStack.back();
    if (evaluationList->empty() || !evaluationList->back().isEndStmt()) {
      const auto &endStmt =
          pftParentStack.back().get<lower::pft::FunctionLikeUnit>().endStmt;
      endStmt.visit(common::visitors{
          [&](const parser::Statement<parser::EndProgramStmt> &s) {
            addEvaluation(lower::pft::Evaluation{
                s.statement, pftParentStack.back(), s.source, s.label});
          },
          [&](const parser::Statement<parser::EndFunctionStmt> &s) {
            addEvaluation(lower::pft::Evaluation{
                s.statement, pftParentStack.back(), s.source, s.label});
          },
          [&](const parser::Statement<parser::EndSubroutineStmt> &s) {
            addEvaluation(lower::pft::Evaluation{
                s.statement, pftParentStack.back(), s.source, s.label});
          },
          [&](const parser::Statement<parser::EndMpSubprogramStmt> &s) {
            addEvaluation(lower::pft::Evaluation{
                s.statement, pftParentStack.back(), s.source, s.label});
          },
          [&](const auto &s) {
            llvm::report_fatal_error("missing end statement or unexpected "
                                     "begin statement reference");
          },
      });
    }
    lastLexicalEvaluation = nullptr;
  }

  /// Pop the ModuleLikeUnit evaluationList when entering the first module
  /// procedure.
  void cleanModuleEvaluationList() {
    if (evaluationListStack.empty())
      return;
    if (pftParentStack.back().isA<lower::pft::ModuleLikeUnit>())
      popEvaluationList();
  }

  /// Initialize a new function-like unit and make it the builder's focus.
  template <typename A>
  bool enterFunction(const A &func,
                     const semantics::SemanticsContext &semanticsContext) {
    cleanModuleEvaluationList();
    endFunctionBody(); // enclosing host subprogram body, if any
    Fortran::lower::pft::FunctionLikeUnit &unit =
        addFunction(lower::pft::FunctionLikeUnit{func, pftParentStack.back(),
                                                 semanticsContext});
    labelEvaluationMap = &unit.labelEvaluationMap;
    assignSymbolLabelMap = &unit.assignSymbolLabelMap;
    functionList = &unit.nestedFunctions;
    pushEvaluationList(&unit.evaluationList);
    pftParentStack.emplace_back(unit);
    return true;
  }

  void exitFunction() {
    rewriteIfGotos();
    endFunctionBody();
    analyzeBranches(nullptr, *evaluationListStack.back()); // add branch links
    processEntryPoints();
    popEvaluationList();
    labelEvaluationMap = nullptr;
    assignSymbolLabelMap = nullptr;
    pftParentStack.pop_back();
    resetFunctionState();
  }

  /// Initialize a new construct or directive and make it the builder's focus.
  template <typename A>
  bool enterConstructOrDirective(const A &constructOrDirective) {
    Fortran::lower::pft::Evaluation &eval = addEvaluation(
        lower::pft::Evaluation{constructOrDirective, pftParentStack.back()});
    eval.evaluationList.reset(new lower::pft::EvaluationList);
    pushEvaluationList(eval.evaluationList.get());
    pftParentStack.emplace_back(eval);
    constructAndDirectiveStack.emplace_back(&eval);
    return true;
  }

  void exitConstructOrDirective() {
    rewriteIfGotos();
    auto *eval = constructAndDirectiveStack.back();
    if (eval->isExecutableDirective()) {
      // A construct at the end of an (unstructured) OpenACC or OpenMP
      // construct region must have an exit target inside the region.
      Fortran::lower::pft::EvaluationList &evaluationList =
          *eval->evaluationList;
      if (!evaluationList.empty() && evaluationList.back().isConstruct()) {
        static const parser::ContinueStmt exitTarget{};
        addEvaluation(
            lower::pft::Evaluation{exitTarget, pftParentStack.back(), {}, {}});
      }
    }
    popEvaluationList();
    pftParentStack.pop_back();
    constructAndDirectiveStack.pop_back();
  }

  /// Reset function state to that of an enclosing host function.
  void resetFunctionState() {
    if (!pftParentStack.empty()) {
      pftParentStack.back().visit(common::visitors{
          [&](lower::pft::FunctionLikeUnit &p) {
            functionList = &p.nestedFunctions;
            labelEvaluationMap = &p.labelEvaluationMap;
            assignSymbolLabelMap = &p.assignSymbolLabelMap;
          },
          [&](lower::pft::ModuleLikeUnit &p) {
            functionList = &p.nestedFunctions;
          },
          [&](auto &) { functionList = nullptr; },
      });
    }
  }

  template <typename A>
  A &addUnit(A &&unit) {
    pgm->getUnits().emplace_back(std::move(unit));
    return std::get<A>(pgm->getUnits().back());
  }

  template <typename A>
  A &addFunction(A &&func) {
    if (functionList) {
      functionList->emplace_back(std::move(func));
      return functionList->back();
    }
    return addUnit(std::move(func));
  }

  // ActionStmt has a couple of non-conforming cases, explicitly handled here.
  // The other cases use an Indirection, which are discarded in the PFT.
  lower::pft::Evaluation
  makeEvaluationAction(const parser::ActionStmt &statement,
                       parser::CharBlock position,
                       std::optional<parser::Label> label) {
    return std::visit(
        common::visitors{
            [&](const auto &x) {
              return lower::pft::Evaluation{
                  removeIndirection(x), pftParentStack.back(), position, label};
            },
        },
        statement.u);
  }

  /// Append an Evaluation to the end of the current list.
  lower::pft::Evaluation &addEvaluation(lower::pft::Evaluation &&eval) {
    assert(functionList && "not in a function");
    assert(!evaluationListStack.empty() && "empty evaluation list stack");
    if (!constructAndDirectiveStack.empty())
      eval.parentConstruct = constructAndDirectiveStack.back();
    auto &entryPointList = eval.getOwningProcedure()->entryPointList;
    evaluationListStack.back()->emplace_back(std::move(eval));
    lower::pft::Evaluation *p = &evaluationListStack.back()->back();
    if (p->isActionStmt() || p->isConstructStmt() || p->isEndStmt() ||
        p->isExecutableDirective()) {
      if (lastLexicalEvaluation) {
        lastLexicalEvaluation->lexicalSuccessor = p;
        p->printIndex = lastLexicalEvaluation->printIndex + 1;
      } else {
        p->printIndex = 1;
      }
      lastLexicalEvaluation = p;
      for (std::size_t entryIndex = entryPointList.size() - 1;
           entryIndex && !entryPointList[entryIndex].second->lexicalSuccessor;
           --entryIndex)
        // Link to the entry's first executable statement.
        entryPointList[entryIndex].second->lexicalSuccessor = p;
    } else if (const auto *entryStmt = p->getIf<parser::EntryStmt>()) {
      const semantics::Symbol *sym =
          std::get<parser::Name>(entryStmt->t).symbol;
      if (auto *details = sym->detailsIf<semantics::GenericDetails>())
        sym = details->specific();
      assert(sym->has<semantics::SubprogramDetails>() &&
             "entry must be a subprogram");
      entryPointList.push_back(std::pair{sym, p});
    }
    if (p->label.has_value())
      labelEvaluationMap->try_emplace(*p->label, p);
    return evaluationListStack.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEvaluationList(lower::pft::EvaluationList *evaluationList) {
    assert(functionList && "not in a function");
    assert(evaluationList && evaluationList->empty() &&
           "evaluation list isn't correct");
    evaluationListStack.emplace_back(evaluationList);
  }

  /// pop the current list and return to the last Evaluation list
  void popEvaluationList() {
    assert(functionList && "not in a function");
    evaluationListStack.pop_back();
  }

  /// Rewrite IfConstructs containing a GotoStmt or CycleStmt to eliminate an
  /// unstructured branch and a trivial basic block.  The pre-branch-analysis
  /// code:
  ///
  ///       <<IfConstruct>>
  ///         1 If[Then]Stmt: if(cond) goto L
  ///         2 GotoStmt: goto L
  ///         3 EndIfStmt
  ///       <<End IfConstruct>>
  ///       4 Statement: ...
  ///       5 Statement: ...
  ///       6 Statement: L ...
  ///
  /// becomes:
  ///
  ///       <<IfConstruct>>
  ///         1 If[Then]Stmt [negate]: if(cond) goto L
  ///         4 Statement: ...
  ///         5 Statement: ...
  ///         3 EndIfStmt
  ///       <<End IfConstruct>>
  ///       6 Statement: L ...
  ///
  /// The If[Then]Stmt condition is implicitly negated.  It is not modified
  /// in the PFT.  It must be negated when generating FIR.  The GotoStmt or
  /// CycleStmt is deleted.
  ///
  /// The transformation is only valid for forward branch targets at the same
  /// construct nesting level as the IfConstruct.  The result must not violate
  /// construct nesting requirements or contain an EntryStmt.  The result
  /// is subject to normal un/structured code classification analysis.  The
  /// result is allowed to violate the F18 Clause 11.1.2.1 prohibition on
  /// transfer of control into the interior of a construct block, as that does
  /// not compromise correct code generation.  When two transformation
  /// candidates overlap, at least one must be disallowed.  In such cases,
  /// the current heuristic favors simple code generation, which happens to
  /// favor later candidates over earlier candidates.  That choice is probably
  /// not significant, but could be changed.
  ///
  void rewriteIfGotos() {
    auto &evaluationList = *evaluationListStack.back();
    if (!evaluationList.size())
      return;
    struct T {
      lower::pft::EvaluationList::iterator ifConstructIt;
      parser::Label ifTargetLabel;
      bool isCycleStmt = false;
    };
    llvm::SmallVector<T> ifCandidateStack;
    const auto *doStmt =
        evaluationList.begin()->getIf<parser::NonLabelDoStmt>();
    std::string doName = doStmt ? getConstructName(*doStmt) : std::string{};
    for (auto it = evaluationList.begin(), end = evaluationList.end();
         it != end; ++it) {
      auto &eval = *it;
      if (eval.isA<parser::EntryStmt>() || eval.isIntermediateConstructStmt()) {
        ifCandidateStack.clear();
        continue;
      }
      auto firstStmt = [](lower::pft::Evaluation *e) {
        return e->isConstruct() ? &*e->evaluationList->begin() : e;
      };
      const Fortran::lower::pft::Evaluation &targetEval = *firstStmt(&eval);
      bool targetEvalIsEndDoStmt = targetEval.isA<parser::EndDoStmt>();
      auto branchTargetMatch = [&]() {
        if (const parser::Label targetLabel =
                ifCandidateStack.back().ifTargetLabel)
          if (targetLabel == *targetEval.label)
            return true; // goto target match
        if (targetEvalIsEndDoStmt && ifCandidateStack.back().isCycleStmt)
          return true; // cycle target match
        return false;
      };
      if (targetEval.label || targetEvalIsEndDoStmt) {
        while (!ifCandidateStack.empty() && branchTargetMatch()) {
          lower::pft::EvaluationList::iterator ifConstructIt =
              ifCandidateStack.back().ifConstructIt;
          lower::pft::EvaluationList::iterator successorIt =
              std::next(ifConstructIt);
          if (successorIt != it) {
            Fortran::lower::pft::EvaluationList &ifBodyList =
                *ifConstructIt->evaluationList;
            lower::pft::EvaluationList::iterator branchStmtIt =
                std::next(ifBodyList.begin());
            assert((branchStmtIt->isA<parser::GotoStmt>() ||
                    branchStmtIt->isA<parser::CycleStmt>()) &&
                   "expected goto or cycle statement");
            ifBodyList.erase(branchStmtIt);
            lower::pft::Evaluation &ifStmt = *ifBodyList.begin();
            ifStmt.negateCondition = true;
            ifStmt.lexicalSuccessor = firstStmt(&*successorIt);
            lower::pft::EvaluationList::iterator endIfStmtIt =
                std::prev(ifBodyList.end());
            std::prev(it)->lexicalSuccessor = &*endIfStmtIt;
            endIfStmtIt->lexicalSuccessor = firstStmt(&*it);
            ifBodyList.splice(endIfStmtIt, evaluationList, successorIt, it);
            for (; successorIt != endIfStmtIt; ++successorIt)
              successorIt->parentConstruct = &*ifConstructIt;
          }
          ifCandidateStack.pop_back();
        }
      }
      if (eval.isA<parser::IfConstruct>() && eval.evaluationList->size() == 3) {
        const auto bodyEval = std::next(eval.evaluationList->begin());
        if (const auto *gotoStmt = bodyEval->getIf<parser::GotoStmt>()) {
          ifCandidateStack.push_back({it, gotoStmt->v});
        } else if (doStmt) {
          if (const auto *cycleStmt = bodyEval->getIf<parser::CycleStmt>()) {
            std::string cycleName = getConstructName(*cycleStmt);
            if (cycleName.empty() || cycleName == doName)
              // This candidate will match doStmt's EndDoStmt.
              ifCandidateStack.push_back({it, {}, true});
          }
        }
      }
    }
  }

  /// Mark IO statement ERR, EOR, and END specifier branch targets.
  /// Mark an IO statement with an assigned format as unstructured.
  template <typename A>
  void analyzeIoBranches(lower::pft::Evaluation &eval, const A &stmt) {
    auto analyzeFormatSpec = [&](const parser::Format &format) {
      if (const auto *expr = std::get_if<parser::Expr>(&format.u)) {
        if (semantics::ExprHasTypeCategory(*semantics::GetExpr(*expr),
                                           common::TypeCategory::Integer))
          eval.isUnstructured = true;
      }
    };
    auto analyzeSpecs{[&](const auto &specList) {
      for (const auto &spec : specList) {
        std::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::Format &format) {
                  analyzeFormatSpec(format);
                },
                [&](const auto &label) {
                  using LabelNodes =
                      std::tuple<parser::ErrLabel, parser::EorLabel,
                                 parser::EndLabel>;
                  if constexpr (common::HasMember<decltype(label), LabelNodes>)
                    markBranchTarget(eval, label.v);
                }},
            spec.u);
      }
    }};

    using OtherIOStmts =
        std::tuple<parser::BackspaceStmt, parser::CloseStmt,
                   parser::EndfileStmt, parser::FlushStmt, parser::OpenStmt,
                   parser::RewindStmt, parser::WaitStmt>;

    if constexpr (std::is_same_v<A, parser::ReadStmt> ||
                  std::is_same_v<A, parser::WriteStmt>) {
      if (stmt.format)
        analyzeFormatSpec(*stmt.format);
      analyzeSpecs(stmt.controls);
    } else if constexpr (std::is_same_v<A, parser::PrintStmt>) {
      analyzeFormatSpec(std::get<parser::Format>(stmt.t));
    } else if constexpr (std::is_same_v<A, parser::InquireStmt>) {
      if (const auto *specList =
              std::get_if<std::list<parser::InquireSpec>>(&stmt.u))
        analyzeSpecs(*specList);
    } else if constexpr (common::HasMember<A, OtherIOStmts>) {
      analyzeSpecs(stmt.v);
    } else {
      // Always crash if this is instantiated
      static_assert(!std::is_same_v<A, parser::ReadStmt>,
                    "Unexpected IO statement");
    }
  }

  /// Set the exit of a construct, possibly from multiple enclosing constructs.
  void setConstructExit(lower::pft::Evaluation &eval) {
    eval.constructExit = &eval.evaluationList->back().nonNopSuccessor();
  }

  /// Mark the target of a branch as a new block.
  void markBranchTarget(lower::pft::Evaluation &sourceEvaluation,
                        lower::pft::Evaluation &targetEvaluation) {
    sourceEvaluation.isUnstructured = true;
    if (!sourceEvaluation.controlSuccessor)
      sourceEvaluation.controlSuccessor = &targetEvaluation;
    targetEvaluation.isNewBlock = true;
    // If this is a branch into the body of a construct (usually illegal,
    // but allowed in some legacy cases), then the targetEvaluation and its
    // ancestors must be marked as unstructured.
    lower::pft::Evaluation *sourceConstruct = sourceEvaluation.parentConstruct;
    lower::pft::Evaluation *targetConstruct = targetEvaluation.parentConstruct;
    if (targetConstruct &&
        &targetConstruct->getFirstNestedEvaluation() == &targetEvaluation)
      // A branch to an initial constructStmt is a branch to the construct.
      targetConstruct = targetConstruct->parentConstruct;
    if (targetConstruct) {
      while (sourceConstruct && sourceConstruct != targetConstruct)
        sourceConstruct = sourceConstruct->parentConstruct;
      if (sourceConstruct != targetConstruct) // branch into a construct body
        for (lower::pft::Evaluation *eval = &targetEvaluation; eval;
             eval = eval->parentConstruct) {
          eval->isUnstructured = true;
          // If the branch is a backward branch into an already analyzed
          // DO or IF construct, mark the construct exit as a new block.
          // For a forward branch, the isUnstructured flag will cause this
          // to be done when the construct is analyzed.
          if (eval->constructExit && (eval->isA<parser::DoConstruct>() ||
                                      eval->isA<parser::IfConstruct>()))
            eval->constructExit->isNewBlock = true;
        }
    }
  }
  void markBranchTarget(lower::pft::Evaluation &sourceEvaluation,
                        parser::Label label) {
    assert(label && "missing branch target label");
    lower::pft::Evaluation *targetEvaluation{
        labelEvaluationMap->find(label)->second};
    assert(targetEvaluation && "missing branch target evaluation");
    markBranchTarget(sourceEvaluation, *targetEvaluation);
  }

  /// Mark the successor of an Evaluation as a new block.
  void markSuccessorAsNewBlock(lower::pft::Evaluation &eval) {
    eval.nonNopSuccessor().isNewBlock = true;
  }

  template <typename A>
  inline std::string getConstructName(const A &stmt) {
    using MaybeConstructNameWrapper =
        std::tuple<parser::BlockStmt, parser::CycleStmt, parser::ElseStmt,
                   parser::ElsewhereStmt, parser::EndAssociateStmt,
                   parser::EndBlockStmt, parser::EndCriticalStmt,
                   parser::EndDoStmt, parser::EndForallStmt, parser::EndIfStmt,
                   parser::EndSelectStmt, parser::EndWhereStmt,
                   parser::ExitStmt>;
    if constexpr (common::HasMember<A, MaybeConstructNameWrapper>) {
      if (stmt.v)
        return stmt.v->ToString();
    }

    using MaybeConstructNameInTuple = std::tuple<
        parser::AssociateStmt, parser::CaseStmt, parser::ChangeTeamStmt,
        parser::CriticalStmt, parser::ElseIfStmt, parser::EndChangeTeamStmt,
        parser::ForallConstructStmt, parser::IfThenStmt, parser::LabelDoStmt,
        parser::MaskedElsewhereStmt, parser::NonLabelDoStmt,
        parser::SelectCaseStmt, parser::SelectRankCaseStmt,
        parser::TypeGuardStmt, parser::WhereConstructStmt>;
    if constexpr (common::HasMember<A, MaybeConstructNameInTuple>) {
      if (auto name = std::get<std::optional<parser::Name>>(stmt.t))
        return name->ToString();
    }

    // These statements have multiple std::optional<parser::Name> elements.
    if constexpr (std::is_same_v<A, parser::SelectRankStmt> ||
                  std::is_same_v<A, parser::SelectTypeStmt>) {
      if (auto name = std::get<0>(stmt.t))
        return name->ToString();
    }

    return {};
  }

  /// \p parentConstruct can be null if this statement is at the highest
  /// level of a program.
  template <typename A>
  void insertConstructName(const A &stmt,
                           lower::pft::Evaluation *parentConstruct) {
    std::string name = getConstructName(stmt);
    if (!name.empty())
      constructNameMap[name] = parentConstruct;
  }

  /// Insert branch links for a list of Evaluations.
  /// \p parentConstruct can be null if the evaluationList contains the
  /// top-level statements of a program.
  void analyzeBranches(lower::pft::Evaluation *parentConstruct,
                       std::list<lower::pft::Evaluation> &evaluationList) {
    lower::pft::Evaluation *lastConstructStmtEvaluation{};
    for (auto &eval : evaluationList) {
      eval.visit(common::visitors{
          // Action statements (except IO statements)
          [&](const parser::CallStmt &s) {
            // Look for alternate return specifiers.
            const auto &args =
                std::get<std::list<parser::ActualArgSpec>>(s.v.t);
            for (const auto &arg : args) {
              const auto &actual = std::get<parser::ActualArg>(arg.t);
              if (const auto *altReturn =
                      std::get_if<parser::AltReturnSpec>(&actual.u))
                markBranchTarget(eval, altReturn->v);
            }
          },
          [&](const parser::CycleStmt &s) {
            std::string name = getConstructName(s);
            lower::pft::Evaluation *construct{name.empty()
                                                  ? doConstructStack.back()
                                                  : constructNameMap[name]};
            assert(construct && "missing CYCLE construct");
            markBranchTarget(eval, construct->evaluationList->back());
          },
          [&](const parser::ExitStmt &s) {
            std::string name = getConstructName(s);
            lower::pft::Evaluation *construct{name.empty()
                                                  ? doConstructStack.back()
                                                  : constructNameMap[name]};
            assert(construct && "missing EXIT construct");
            markBranchTarget(eval, *construct->constructExit);
          },
          [&](const parser::FailImageStmt &) {
            eval.isUnstructured = true;
            if (eval.lexicalSuccessor->lexicalSuccessor)
              markSuccessorAsNewBlock(eval);
          },
          [&](const parser::GotoStmt &s) { markBranchTarget(eval, s.v); },
          [&](const parser::IfStmt &) {
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::ReturnStmt &) {
            eval.isUnstructured = true;
            if (eval.lexicalSuccessor->lexicalSuccessor)
              markSuccessorAsNewBlock(eval);
          },
          [&](const parser::StopStmt &) {
            eval.isUnstructured = true;
            if (eval.lexicalSuccessor->lexicalSuccessor)
              markSuccessorAsNewBlock(eval);
          },
          [&](const parser::ComputedGotoStmt &s) {
            for (auto &label : std::get<std::list<parser::Label>>(s.t))
              markBranchTarget(eval, label);
          },
          [&](const parser::ArithmeticIfStmt &s) {
            markBranchTarget(eval, std::get<1>(s.t));
            markBranchTarget(eval, std::get<2>(s.t));
            markBranchTarget(eval, std::get<3>(s.t));
          },
          [&](const parser::AssignStmt &s) { // legacy label assignment
            auto &label = std::get<parser::Label>(s.t);
            const auto *sym = std::get<parser::Name>(s.t).symbol;
            assert(sym && "missing AssignStmt symbol");
            lower::pft::Evaluation *target{
                labelEvaluationMap->find(label)->second};
            assert(target && "missing branch target evaluation");
            if (!target->isA<parser::FormatStmt>())
              target->isNewBlock = true;
            auto iter = assignSymbolLabelMap->find(*sym);
            if (iter == assignSymbolLabelMap->end()) {
              lower::pft::LabelSet labelSet{};
              labelSet.insert(label);
              assignSymbolLabelMap->try_emplace(*sym, labelSet);
            } else {
              iter->second.insert(label);
            }
          },
          [&](const parser::AssignedGotoStmt &) {
            // Although this statement is a branch, it doesn't have any
            // explicit control successors.  So the code at the end of the
            // loop won't mark the successor.  Do that here.
            eval.isUnstructured = true;
            markSuccessorAsNewBlock(eval);
          },

          // The first executable statement after an EntryStmt is a new block.
          [&](const parser::EntryStmt &) {
            eval.lexicalSuccessor->isNewBlock = true;
          },

          // Construct statements
          [&](const parser::AssociateStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::BlockStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::SelectCaseStmt &s) {
            insertConstructName(s, parentConstruct);
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::CaseStmt &) {
            eval.isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::EndSelectStmt &) {
            eval.nonNopSuccessor().isNewBlock = true;
            lastConstructStmtEvaluation = nullptr;
          },
          [&](const parser::ChangeTeamStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::CriticalStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::NonLabelDoStmt &s) {
            insertConstructName(s, parentConstruct);
            doConstructStack.push_back(parentConstruct);
            const auto &loopControl =
                std::get<std::optional<parser::LoopControl>>(s.t);
            if (!loopControl.has_value()) {
              eval.isUnstructured = true; // infinite loop
              return;
            }
            eval.nonNopSuccessor().isNewBlock = true;
            eval.controlSuccessor = &evaluationList.back();
            if (const auto *bounds =
                    std::get_if<parser::LoopControl::Bounds>(&loopControl->u)) {
              if (bounds->name.thing.symbol->GetType()->IsNumeric(
                      common::TypeCategory::Real))
                eval.isUnstructured = true; // real-valued loop control
            } else if (std::get_if<parser::ScalarLogicalExpr>(
                           &loopControl->u)) {
              eval.isUnstructured = true; // while loop
            }
          },
          [&](const parser::EndDoStmt &) {
            lower::pft::Evaluation &doEval = evaluationList.front();
            eval.controlSuccessor = &doEval;
            doConstructStack.pop_back();
            if (parentConstruct->lowerAsStructured())
              return;
            // The loop is unstructured, which wasn't known for all cases when
            // visiting the NonLabelDoStmt.
            parentConstruct->constructExit->isNewBlock = true;
            const auto &doStmt = *doEval.getIf<parser::NonLabelDoStmt>();
            const auto &loopControl =
                std::get<std::optional<parser::LoopControl>>(doStmt.t);
            if (!loopControl.has_value())
              return; // infinite loop
            if (const auto *concurrent =
                    std::get_if<parser::LoopControl::Concurrent>(
                        &loopControl->u)) {
              // If there is a mask, the EndDoStmt starts a new block.
              const auto &header =
                  std::get<parser::ConcurrentHeader>(concurrent->t);
              eval.isNewBlock |=
                  std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)
                      .has_value();
            }
          },
          [&](const parser::IfThenStmt &s) {
            insertConstructName(s, parentConstruct);
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::ElseIfStmt &) {
            eval.isNewBlock = true;
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::ElseStmt &) {
            eval.isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = nullptr;
          },
          [&](const parser::EndIfStmt &) {
            if (parentConstruct->lowerAsUnstructured())
              parentConstruct->constructExit->isNewBlock = true;
            if (lastConstructStmtEvaluation) {
              lastConstructStmtEvaluation->controlSuccessor =
                  parentConstruct->constructExit;
              lastConstructStmtEvaluation = nullptr;
            }
          },
          [&](const parser::SelectRankStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::SelectRankCaseStmt &) { eval.isNewBlock = true; },
          [&](const parser::SelectTypeStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::TypeGuardStmt &) { eval.isNewBlock = true; },

          // Constructs - set (unstructured) construct exit targets
          [&](const parser::AssociateConstruct &) { setConstructExit(eval); },
          [&](const parser::BlockConstruct &) {
            // EndBlockStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::CaseConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },
          [&](const parser::ChangeTeamConstruct &) {
            // EndChangeTeamStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::CriticalConstruct &) {
            // EndCriticalStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::DoConstruct &) { setConstructExit(eval); },
          [&](const parser::ForallConstruct &) { setConstructExit(eval); },
          [&](const parser::IfConstruct &) { setConstructExit(eval); },
          [&](const parser::SelectRankConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },
          [&](const parser::SelectTypeConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },
          [&](const parser::WhereConstruct &) { setConstructExit(eval); },

          // Default - Common analysis for IO statements; otherwise nop.
          [&](const auto &stmt) {
            using A = std::decay_t<decltype(stmt)>;
            using IoStmts = std::tuple<
                parser::BackspaceStmt, parser::CloseStmt, parser::EndfileStmt,
                parser::FlushStmt, parser::InquireStmt, parser::OpenStmt,
                parser::PrintStmt, parser::ReadStmt, parser::RewindStmt,
                parser::WaitStmt, parser::WriteStmt>;
            if constexpr (common::HasMember<A, IoStmts>)
              analyzeIoBranches(eval, stmt);
          },
      });

      // Analyze construct evaluations.
      if (eval.evaluationList)
        analyzeBranches(&eval, *eval.evaluationList);

      // Set the successor of the last statement in an IF or SELECT block.
      if (!eval.controlSuccessor && eval.lexicalSuccessor &&
          eval.lexicalSuccessor->isIntermediateConstructStmt()) {
        eval.controlSuccessor = parentConstruct->constructExit;
        eval.lexicalSuccessor->isNewBlock = true;
      }

      // Propagate isUnstructured flag to enclosing construct.
      if (parentConstruct && eval.isUnstructured)
        parentConstruct->isUnstructured = true;

      // The successor of a branch starts a new block.
      if (eval.controlSuccessor && eval.isActionStmt() &&
          eval.lowerAsUnstructured())
        markSuccessorAsNewBlock(eval);
    }
  }

  /// Do processing specific to subprograms with multiple entry points.
  void processEntryPoints() {
    lower::pft::Evaluation *initialEval = &evaluationListStack.back()->front();
    lower::pft::FunctionLikeUnit *unit = initialEval->getOwningProcedure();
    int entryCount = unit->entryPointList.size();
    if (entryCount == 1)
      return;

    // The first executable statement in the subprogram is preceded by a
    // branch to the entry point, so it starts a new block.
    if (initialEval->hasNestedEvaluations())
      initialEval = &initialEval->getFirstNestedEvaluation();
    else if (initialEval->isA<Fortran::parser::EntryStmt>())
      initialEval = initialEval->lexicalSuccessor;
    initialEval->isNewBlock = true;

    // All function entry points share a single result container.
    // Find one of the largest results.
    for (int entryIndex = 0; entryIndex < entryCount; ++entryIndex) {
      unit->setActiveEntry(entryIndex);
      const auto &details =
          unit->getSubprogramSymbol().get<semantics::SubprogramDetails>();
      if (details.isFunction()) {
        const semantics::Symbol *resultSym = &details.result();
        assert(resultSym && "missing result symbol");
        if (!unit->primaryResult ||
            unit->primaryResult->size() < resultSym->size())
          unit->primaryResult = resultSym;
      }
    }
    unit->setActiveEntry(0);
  }

  std::unique_ptr<lower::pft::Program> pgm;
  std::vector<lower::pft::PftNode> pftParentStack;
  const semantics::SemanticsContext &semanticsContext;

  /// functionList points to the internal or module procedure function list
  /// of a FunctionLikeUnit or a ModuleLikeUnit.  It may be null.
  std::list<lower::pft::FunctionLikeUnit> *functionList{};
  std::vector<lower::pft::Evaluation *> constructAndDirectiveStack{};
  std::vector<lower::pft::Evaluation *> doConstructStack{};
  /// evaluationListStack is the current nested construct evaluationList state.
  std::vector<lower::pft::EvaluationList *> evaluationListStack{};
  llvm::DenseMap<parser::Label, lower::pft::Evaluation *> *labelEvaluationMap{};
  lower::pft::SymbolLabelMap *assignSymbolLabelMap{};
  std::map<std::string, lower::pft::Evaluation *> constructNameMap{};
  lower::pft::Evaluation *lastLexicalEvaluation{};
};

class PFTDumper {
public:
  void dumpPFT(llvm::raw_ostream &outputStream,
               const lower::pft::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(common::visitors{
                     [&](const lower::pft::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEnd BlockData\n\n";
                     },
                     [&](const lower::pft::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](const lower::pft::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                     [&](const lower::pft::CompilerDirectiveUnit &unit) {
                       dumpCompilerDirectiveUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
  }

  llvm::StringRef evaluationName(const lower::pft::Evaluation &eval) {
    return eval.visit([](const auto &parseTreeNode) {
      return parser::ParseTreeDumper::GetNodeName(parseTreeNode);
    });
  }

  void dumpEvaluation(llvm::raw_ostream &outputStream,
                      const lower::pft::Evaluation &eval,
                      const std::string &indentString, int indent = 1) {
    llvm::StringRef name = evaluationName(eval);
    llvm::StringRef newBlock = eval.isNewBlock ? "^" : "";
    llvm::StringRef bang = eval.isUnstructured ? "!" : "";
    outputStream << indentString;
    if (eval.printIndex)
      outputStream << eval.printIndex << ' ';
    if (eval.hasNestedEvaluations())
      outputStream << "<<" << newBlock << name << bang << ">>";
    else
      outputStream << newBlock << name << bang;
    if (eval.negateCondition)
      outputStream << " [negate]";
    if (eval.constructExit)
      outputStream << " -> " << eval.constructExit->printIndex;
    else if (eval.controlSuccessor)
      outputStream << " -> " << eval.controlSuccessor->printIndex;
    else if (eval.isA<parser::EntryStmt>() && eval.lexicalSuccessor)
      outputStream << " -> " << eval.lexicalSuccessor->printIndex;
    if (!eval.position.empty())
      outputStream << ": " << eval.position.ToString();
    else if (auto *dir = eval.getIf<Fortran::parser::CompilerDirective>())
      outputStream << ": !" << dir->source.ToString();
    outputStream << '\n';
    if (eval.hasNestedEvaluations()) {
      dumpEvaluationList(outputStream, *eval.evaluationList, indent + 1);
      outputStream << indentString << "<<End " << name << bang << ">>\n";
    }
  }

  void dumpEvaluation(llvm::raw_ostream &ostream,
                      const lower::pft::Evaluation &eval) {
    dumpEvaluation(ostream, eval, "");
  }

  void dumpEvaluationList(llvm::raw_ostream &outputStream,
                          const lower::pft::EvaluationList &evaluationList,
                          int indent = 1) {
    static const auto white = "                                      ++"s;
    auto indentString = white.substr(0, indent * 2);
    for (const lower::pft::Evaluation &eval : evaluationList)
      dumpEvaluation(outputStream, eval, indentString, indent);
  }

  void
  dumpFunctionLikeUnit(llvm::raw_ostream &outputStream,
                       const lower::pft::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    llvm::StringRef unitKind;
    llvm::StringRef name;
    llvm::StringRef header;
    if (functionLikeUnit.beginStmt) {
      functionLikeUnit.beginStmt->visit(common::visitors{
          [&](const parser::Statement<parser::ProgramStmt> &stmt) {
            unitKind = "Program";
            name = toStringRef(stmt.statement.v.source);
          },
          [&](const parser::Statement<parser::FunctionStmt> &stmt) {
            unitKind = "Function";
            name = toStringRef(std::get<parser::Name>(stmt.statement.t).source);
            header = toStringRef(stmt.source);
          },
          [&](const parser::Statement<parser::SubroutineStmt> &stmt) {
            unitKind = "Subroutine";
            name = toStringRef(std::get<parser::Name>(stmt.statement.t).source);
            header = toStringRef(stmt.source);
          },
          [&](const parser::Statement<parser::MpSubprogramStmt> &stmt) {
            unitKind = "MpSubprogram";
            name = toStringRef(stmt.statement.v.source);
            header = toStringRef(stmt.source);
          },
          [&](const auto &) { llvm_unreachable("not a valid begin stmt"); },
      });
    } else {
      unitKind = "Program";
      name = "<anonymous>";
    }
    outputStream << unitKind << ' ' << name;
    if (!header.empty())
      outputStream << ": " << header;
    outputStream << '\n';
    dumpEvaluationList(outputStream, functionLikeUnit.evaluationList);
    if (!functionLikeUnit.nestedFunctions.empty()) {
      outputStream << "\nContains\n";
      for (const lower::pft::FunctionLikeUnit &func :
           functionLikeUnit.nestedFunctions)
        dumpFunctionLikeUnit(outputStream, func);
      outputStream << "End Contains\n";
    }
    outputStream << "End " << unitKind << ' ' << name << "\n\n";
  }

  void dumpModuleLikeUnit(llvm::raw_ostream &outputStream,
                          const lower::pft::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike:\n";
    dumpEvaluationList(outputStream, moduleLikeUnit.evaluationList);
    outputStream << "Contains\n";
    for (const lower::pft::FunctionLikeUnit &func :
         moduleLikeUnit.nestedFunctions)
      dumpFunctionLikeUnit(outputStream, func);
    outputStream << "End Contains\nEnd ModuleLike\n\n";
  }

  // Top level directives
  void dumpCompilerDirectiveUnit(
      llvm::raw_ostream &outputStream,
      const lower::pft::CompilerDirectiveUnit &directive) {
    outputStream << getNodeIndex(directive) << " ";
    outputStream << "CompilerDirective: !";
    outputStream << directive.get<Fortran::parser::CompilerDirective>()
                        .source.ToString();
    outputStream << "\nEnd CompilerDirective\n\n";
  }

  template <typename T>
  std::size_t getNodeIndex(const T &node) {
    auto addr = static_cast<const void *>(&node);
    auto it = nodeIndexes.find(addr);
    if (it != nodeIndexes.end())
      return it->second;
    nodeIndexes.try_emplace(addr, nextIndex);
    return nextIndex++;
  }
  std::size_t getNodeIndex(const lower::pft::Program &) { return 0; }

private:
  llvm::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

} // namespace

template <typename A, typename T>
static lower::pft::FunctionLikeUnit::FunctionStatement
getFunctionStmt(const T &func) {
  lower::pft::FunctionLikeUnit::FunctionStatement result{
      std::get<parser::Statement<A>>(func.t)};
  return result;
}

template <typename A, typename T>
static lower::pft::ModuleLikeUnit::ModuleStatement getModuleStmt(const T &mod) {
  lower::pft::ModuleLikeUnit::ModuleStatement result{
      std::get<parser::Statement<A>>(mod.t)};
  return result;
}

template <typename A>
static const semantics::Symbol *getSymbol(A &beginStmt) {
  const auto *symbol = beginStmt.visit(common::visitors{
      [](const parser::Statement<parser::ProgramStmt> &stmt)
          -> const semantics::Symbol * { return stmt.statement.v.symbol; },
      [](const parser::Statement<parser::FunctionStmt> &stmt)
          -> const semantics::Symbol * {
        return std::get<parser::Name>(stmt.statement.t).symbol;
      },
      [](const parser::Statement<parser::SubroutineStmt> &stmt)
          -> const semantics::Symbol * {
        return std::get<parser::Name>(stmt.statement.t).symbol;
      },
      [](const parser::Statement<parser::MpSubprogramStmt> &stmt)
          -> const semantics::Symbol * { return stmt.statement.v.symbol; },
      [](const parser::Statement<parser::ModuleStmt> &stmt)
          -> const semantics::Symbol * { return stmt.statement.v.symbol; },
      [](const parser::Statement<parser::SubmoduleStmt> &stmt)
          -> const semantics::Symbol * {
        return std::get<parser::Name>(stmt.statement.t).symbol;
      },
      [](const auto &) -> const semantics::Symbol * {
        llvm_unreachable("unknown FunctionLike or ModuleLike beginStmt");
        return nullptr;
      }});
  assert(symbol && "parser::Name must have resolved symbol");
  return symbol;
}

bool Fortran::lower::pft::Evaluation::lowerAsStructured() const {
  return !lowerAsUnstructured();
}

bool Fortran::lower::pft::Evaluation::lowerAsUnstructured() const {
  return isUnstructured || clDisableStructuredFir;
}

lower::pft::FunctionLikeUnit *
Fortran::lower::pft::Evaluation::getOwningProcedure() const {
  return parent.visit(common::visitors{
      [](lower::pft::FunctionLikeUnit &c) { return &c; },
      [&](lower::pft::Evaluation &c) { return c.getOwningProcedure(); },
      [](auto &) -> lower::pft::FunctionLikeUnit * { return nullptr; },
  });
}

bool Fortran::lower::definedInCommonBlock(const semantics::Symbol &sym) {
  return semantics::FindCommonBlockContaining(sym);
}

static bool isReentrant(const Fortran::semantics::Scope &scope) {
  if (scope.kind() == Fortran::semantics::Scope::Kind::MainProgram)
    return false;
  if (scope.kind() == Fortran::semantics::Scope::Kind::Subprogram) {
    const Fortran::semantics::Symbol *sym = scope.symbol();
    assert(sym && "Subprogram scope must have a symbol");
    return sym->attrs().test(semantics::Attr::RECURSIVE) ||
           (!sym->attrs().test(semantics::Attr::NON_RECURSIVE) &&
            Fortran::lower::defaultRecursiveFunctionSetting());
  }
  if (scope.kind() == Fortran::semantics::Scope::Kind::Module)
    return false;
  return true;
}

/// Is the symbol `sym` a global?
bool Fortran::lower::symbolIsGlobal(const semantics::Symbol &sym) {
  if (const auto *details = sym.detailsIf<semantics::ObjectEntityDetails>()) {
    if (details->init())
      return true;
    if (!isReentrant(sym.owner())) {
      // Turn array and character of non re-entrant programs (like the main
      // program) into global memory.
      if (const Fortran::semantics::DeclTypeSpec *symTy = sym.GetType())
        if (symTy->category() == semantics::DeclTypeSpec::Character)
          if (auto e = symTy->characterTypeSpec().length().GetExplicit())
            return true;
      if (!details->shape().empty() || !details->coshape().empty())
        return true;
    }
    if (mainProgramGlobals &&
        sym.owner().kind() == Fortran::semantics::Scope::Kind::MainProgram)
      return true;
  }
  return semantics::IsSaved(sym) || lower::definedInCommonBlock(sym) ||
         semantics::IsNamedConstant(sym);
}

namespace {
/// This helper class is for sorting the symbols in the symbol table. We want
/// the symbols in an order such that a symbol will be visited after those it
/// depends upon. Otherwise this sort is stable and preserves the order of the
/// symbol table, which is sorted by name.
struct SymbolDependenceDepth {
  explicit SymbolDependenceDepth(
      std::vector<std::vector<lower::pft::Variable>> &vars)
      : vars{vars} {}

  void analyzeAliasesInCurrentScope(const semantics::Scope &scope) {
    // FIXME: When this function is called on the scope of an internal
    // procedure whose parent contains an EQUIVALENCE set and the internal
    // procedure uses variables from that EQUIVALENCE set, we end up creating
    // an AggregateStore for those variables unnecessarily.
    //
    /// If this is a function nested in a module no host associated
    /// symbol are added to the function scope for module symbols used in this
    /// scope. As a result, alias analysis in parent module scopes must be
    /// preformed here.
    const semantics::Scope *parentScope = &scope;
    while (!parentScope->IsGlobal()) {
      parentScope = &parentScope->parent();
      if (parentScope->IsModule())
        analyzeAliases(*parentScope);
    }
    for (const auto &iter : scope) {
      const semantics::Symbol &ultimate = iter.second.get().GetUltimate();
      if (skipSymbol(ultimate))
        continue;
      analyzeAliases(ultimate.owner());
    }
    // add all aggregate stores to the front of the work list
    adjustSize(1);
    // The copy in the loop matters, 'stores' will still be used.
    for (auto st : stores)
      vars[0].emplace_back(std::move(st));
  }

  // Compute the offset of the last byte that resides in the symbol.
  inline static std::size_t offsetWidth(const Fortran::semantics::Symbol &sym) {
    std::size_t width = sym.offset();
    if (std::size_t size = sym.size())
      width += size - 1;
    return width;
  }

  // Analyze the equivalence sets. This analysis need not be performed when the
  // scope has no equivalence sets.
  void analyzeAliases(const semantics::Scope &scope) {
    if (scope.equivalenceSets().empty())
      return;
    // Don't analyze a scope if it has already been analyzed.
    if (analyzedScopes.find(&scope) != analyzedScopes.end())
      return;

    analyzedScopes.insert(&scope);
    std::list<std::list<semantics::SymbolRef>> aggregates =
        Fortran::semantics::GetStorageAssociations(scope);
    for (std::list<semantics::SymbolRef> aggregate : aggregates) {
      const Fortran::semantics::Symbol *aggregateSym = nullptr;
      bool isGlobal = false;
      const semantics::Symbol &first = *aggregate.front();
      std::size_t start = first.offset();
      std::size_t end = first.offset() + first.size();
      const Fortran::semantics::Symbol *namingSym = nullptr;
      for (semantics::SymbolRef symRef : aggregate) {
        const semantics::Symbol &sym = *symRef;
        aliasSyms.insert(&sym);
        if (sym.test(Fortran::semantics::Symbol::Flag::CompilerCreated)) {
          aggregateSym = &sym;
        } else {
          isGlobal |= lower::symbolIsGlobal(sym);
          start = std::min(sym.offset(), start);
          end = std::max(sym.offset() + sym.size(), end);
          if (!namingSym || (sym.name() < namingSym->name()))
            namingSym = &sym;
        }
      }
      assert(namingSym && "must contain at least one user symbol");
      if (!aggregateSym) {
        stores.emplace_back(
            Fortran::lower::pft::Variable::Interval{start, end - start},
            *namingSym, isGlobal);
      } else {
        stores.emplace_back(*aggregateSym, *namingSym, isGlobal);
      }
    }
  }

  // Recursively visit each symbol to determine the height of its dependence on
  // other symbols.
  int analyze(const semantics::Symbol &sym) {
    auto done = seen.insert(&sym);
    LLVM_DEBUG(llvm::dbgs() << "analyze symbol: " << sym << '\n');
    if (!done.second)
      return 0;
    const bool isProcedurePointerOrDummy =
        semantics::IsProcedurePointer(sym) ||
        (semantics::IsProcedure(sym) && IsDummy(sym));
    // A procedure argument in a subprogram with multiple entry points might
    // need a vars list entry to trigger creation of a symbol map entry in
    // some cases.  Non-dummy procedures don't.
    if (semantics::IsProcedure(sym) && !isProcedurePointerOrDummy)
      return 0;
    semantics::Symbol ultimate = sym.GetUltimate();
    if (const auto *details =
            ultimate.detailsIf<semantics::NamelistDetails>()) {
      // handle namelist group symbols
      for (const semantics::SymbolRef &s : details->objects())
        analyze(s);
      return 0;
    }
    if (!ultimate.has<semantics::ObjectEntityDetails>() &&
        !isProcedurePointerOrDummy)
      return 0;

    if (sym.has<semantics::DerivedTypeDetails>())
      llvm_unreachable("not yet implemented - derived type analysis");

    // Symbol must be something lowering will have to allocate.
    int depth = 0;
    // Analyze symbols appearing in object entity specification expression. This
    // ensures these symbols will be instantiated before the current one.
    // This is not done for object entities that are host associated because
    // they must be instantiated from the value of the host symbols (the
    // specification expressions should not be re-evaluated).
    if (const auto *details = sym.detailsIf<semantics::ObjectEntityDetails>()) {
      const semantics::DeclTypeSpec *symTy = sym.GetType();
      assert(symTy && "symbol must have a type");
      // check CHARACTER's length
      if (symTy->category() == semantics::DeclTypeSpec::Character)
        if (auto e = symTy->characterTypeSpec().length().GetExplicit())
          for (const auto &s : evaluate::CollectSymbols(*e))
            depth = std::max(analyze(s) + 1, depth);

      auto doExplicit = [&](const auto &bound) {
        if (bound.isExplicit()) {
          semantics::SomeExpr e{*bound.GetExplicit()};
          for (const auto &s : evaluate::CollectSymbols(e))
            depth = std::max(analyze(s) + 1, depth);
        }
      };
      // handle any symbols in array bound declarations
      for (const semantics::ShapeSpec &subs : details->shape()) {
        doExplicit(subs.lbound());
        doExplicit(subs.ubound());
      }
      // handle any symbols in coarray bound declarations
      for (const semantics::ShapeSpec &subs : details->coshape()) {
        doExplicit(subs.lbound());
        doExplicit(subs.ubound());
      }
      // handle any symbols in initialization expressions
      if (auto e = details->init())
        for (const auto &s : evaluate::CollectSymbols(*e))
          depth = std::max(analyze(s) + 1, depth);
    }
    adjustSize(depth + 1);
    bool global = lower::symbolIsGlobal(sym);
    vars[depth].emplace_back(sym, global, depth);
    if (semantics::IsAllocatable(sym))
      vars[depth].back().setHeapAlloc();
    if (semantics::IsPointer(sym))
      vars[depth].back().setPointer();
    if (ultimate.attrs().test(semantics::Attr::TARGET))
      vars[depth].back().setTarget();

    // If there are alias sets, then link the participating variables to their
    // aggregate stores when constructing the new variable on the list.
    if (lower::pft::Variable::AggregateStore *store = findStoreIfAlias(sym))
      vars[depth].back().setAlias(store->getOffset());
    return depth;
  }

  /// Save the final list of variable allocations as a single vector and free
  /// the rest.
  void finalize() {
    for (int i = 1, end = vars.size(); i < end; ++i)
      vars[0].insert(vars[0].end(), vars[i].begin(), vars[i].end());
    vars.resize(1);
  }

  Fortran::lower::pft::Variable::AggregateStore *
  findStoreIfAlias(const Fortran::evaluate::Symbol &sym) {
    const semantics::Symbol &ultimate = sym.GetUltimate();
    const semantics::Scope &scope = ultimate.owner();
    // Expect the total number of EQUIVALENCE sets to be small for a typical
    // Fortran program.
    if (aliasSyms.find(&ultimate) != aliasSyms.end()) {
      LLVM_DEBUG(llvm::dbgs() << "symbol: " << ultimate << '\n');
      LLVM_DEBUG(llvm::dbgs() << "scope: " << scope << '\n');
      std::size_t off = ultimate.offset();
      std::size_t symSize = ultimate.size();
      for (lower::pft::Variable::AggregateStore &v : stores) {
        if (&v.getOwningScope() == &scope) {
          auto intervalOff = std::get<0>(v.interval);
          auto intervalSize = std::get<1>(v.interval);
          if (off >= intervalOff && off < intervalOff + intervalSize)
            return &v;
          // Zero sized symbol in zero sized equivalence.
          if (off == intervalOff && symSize == 0)
            return &v;
        }
      }
      // clang-format off
      LLVM_DEBUG(
          llvm::dbgs() << "looking for " << off << "\n{\n";
          for (lower::pft::Variable::AggregateStore &v : stores) {
            llvm::dbgs() << " in scope: " << &v.getOwningScope() << "\n";
            llvm::dbgs() << "  i = [" << std::get<0>(v.interval) << ".."
                << std::get<0>(v.interval) + std::get<1>(v.interval)
                << "]\n";
          }
          llvm::dbgs() << "}\n");
      // clang-format on
      llvm_unreachable("the store must be present");
    }
    return nullptr;
  }

private:
  /// Skip symbol in alias analysis.
  bool skipSymbol(const semantics::Symbol &sym) {
    // Common block equivalences are largely managed by the front end.
    // Compiler generated symbols ('.' names) cannot be equivalenced.
    // FIXME: Equivalence code generation may need to be revisited.
    return !sym.has<semantics::ObjectEntityDetails>() ||
           lower::definedInCommonBlock(sym) || sym.name()[0] == '.';
  }

  // Make sure the table is of appropriate size.
  void adjustSize(std::size_t size) {
    if (vars.size() < size)
      vars.resize(size);
  }

  llvm::SmallSet<const semantics::Symbol *, 32> seen;
  std::vector<std::vector<lower::pft::Variable>> &vars;
  llvm::SmallSet<const semantics::Symbol *, 32> aliasSyms;
  /// Set of Scope that have been analyzed for aliases.
  llvm::SmallSet<const semantics::Scope *, 4> analyzedScopes;
  std::vector<Fortran::lower::pft::Variable::AggregateStore> stores;
};
} // namespace

static void processSymbolTable(
    const semantics::Scope &scope,
    std::vector<std::vector<Fortran::lower::pft::Variable>> &varList) {
  SymbolDependenceDepth sdd{varList};
  sdd.analyzeAliasesInCurrentScope(scope);
  for (const auto &iter : scope)
    sdd.analyze(iter.second.get());
  sdd.finalize();
}

//===----------------------------------------------------------------------===//
// FunctionLikeUnit implementation
//===----------------------------------------------------------------------===//

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::MainProgram &func, const lower::pft::PftNode &parent,
    const semantics::SemanticsContext &semanticsContext)
    : ProgramUnit{func, parent}, endStmt{
                                     getFunctionStmt<parser::EndProgramStmt>(
                                         func)} {
  const auto &programStmt =
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(func.t);
  if (programStmt.has_value()) {
    beginStmt = FunctionStatement(programStmt.value());
    const semantics::Symbol *symbol = getSymbol(*beginStmt);
    entryPointList[0].first = symbol;
    processSymbolTable(*symbol->scope(), varList);
  } else {
    processSymbolTable(
        semanticsContext.FindScope(
            std::get<parser::Statement<parser::EndProgramStmt>>(func.t).source),
        varList);
  }
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::FunctionSubprogram &func, const lower::pft::PftNode &parent,
    const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::FunctionStmt>(func)},
      endStmt{getFunctionStmt<parser::EndFunctionStmt>(func)} {
  const semantics::Symbol *symbol = getSymbol(*beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope(), varList);
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SubroutineSubprogram &func, const lower::pft::PftNode &parent,
    const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<parser::EndSubroutineStmt>(func)} {
  const semantics::Symbol *symbol = getSymbol(*beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope(), varList);
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SeparateModuleSubprogram &func,
    const lower::pft::PftNode &parent, const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<parser::EndMpSubprogramStmt>(func)} {
  const semantics::Symbol *symbol = getSymbol(*beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope(), varList);
}

Fortran::lower::HostAssociations &
Fortran::lower::pft::FunctionLikeUnit::parentHostAssoc() {
  if (auto *par = parent.getIf<FunctionLikeUnit>())
    return par->hostAssociations;
  llvm::report_fatal_error("parent is not a function");
}

bool Fortran::lower::pft::FunctionLikeUnit::parentHasHostAssoc() {
  if (auto *par = parent.getIf<FunctionLikeUnit>())
    return !par->hostAssociations.empty();
  return false;
}

parser::CharBlock
Fortran::lower::pft::FunctionLikeUnit::getStartingSourceLoc() const {
  if (beginStmt)
    return stmtSourceLoc(*beginStmt);
  if (!evaluationList.empty())
    return evaluationList.front().position;
  return stmtSourceLoc(endStmt);
}

//===----------------------------------------------------------------------===//
// ModuleLikeUnit implementation
//===----------------------------------------------------------------------===//

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const parser::Module &m, const lower::pft::PftNode &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::ModuleStmt>(m)},
      endStmt{getModuleStmt<parser::EndModuleStmt>(m)} {
  const semantics::Symbol *symbol = getSymbol(beginStmt);
  processSymbolTable(*symbol->scope(), varList);
}

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const parser::Submodule &m, const lower::pft::PftNode &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::SubmoduleStmt>(
                                  m)},
      endStmt{getModuleStmt<parser::EndSubmoduleStmt>(m)} {
  const semantics::Symbol *symbol = getSymbol(beginStmt);
  processSymbolTable(*symbol->scope(), varList);
}

parser::CharBlock
Fortran::lower::pft::ModuleLikeUnit::getStartingSourceLoc() const {
  return stmtSourceLoc(beginStmt);
}
const Fortran::semantics::Scope &
Fortran::lower::pft::ModuleLikeUnit::getScope() const {
  const Fortran::semantics::Symbol *symbol = getSymbol(beginStmt);
  assert(symbol && symbol->scope() &&
         "Module statement must have a symbol with a scope");
  return *symbol->scope();
}

//===----------------------------------------------------------------------===//
// BlockDataUnit implementation
//===----------------------------------------------------------------------===//

Fortran::lower::pft::BlockDataUnit::BlockDataUnit(
    const parser::BlockData &bd, const lower::pft::PftNode &parent,
    const semantics::SemanticsContext &semanticsContext)
    : ProgramUnit{bd, parent},
      symTab{semanticsContext.FindScope(
          std::get<parser::Statement<parser::EndBlockDataStmt>>(bd.t).source)} {
}

std::unique_ptr<lower::pft::Program>
Fortran::lower::createPFT(const parser::Program &root,
                          const semantics::SemanticsContext &semanticsContext) {
  PFTBuilder walker(semanticsContext);
  Walk(root, walker);
  return walker.result();
}

// FIXME: FlangDriver
// This option should be integrated with the real driver as the default of
// RECURSIVE vs. NON_RECURSIVE may be changed by other command line options,
// etc., etc.
bool Fortran::lower::defaultRecursiveFunctionSetting() {
  return !nonRecursiveProcedures;
}

void Fortran::lower::dumpPFT(llvm::raw_ostream &outputStream,
                             const lower::pft::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

void Fortran::lower::pft::Program::dump() const {
  dumpPFT(llvm::errs(), *this);
}

void Fortran::lower::pft::Evaluation::dump() const {
  PFTDumper{}.dumpEvaluation(llvm::errs(), *this);
}

void Fortran::lower::pft::Variable::dump() const {
  if (auto *s = std::get_if<Nominal>(&var)) {
    llvm::errs() << "symbol: " << s->symbol->name();
    llvm::errs() << " (depth: " << s->depth << ')';
    if (s->global)
      llvm::errs() << ", global";
    if (s->heapAlloc)
      llvm::errs() << ", allocatable";
    if (s->pointer)
      llvm::errs() << ", pointer";
    if (s->target)
      llvm::errs() << ", target";
    if (s->aliaser)
      llvm::errs() << ", equivalence(" << s->aliasOffset << ')';
  } else if (auto *s = std::get_if<AggregateStore>(&var)) {
    llvm::errs() << "interval[" << std::get<0>(s->interval) << ", "
                 << std::get<1>(s->interval) << "]:";
    llvm::errs() << " name: " << toStringRef(s->getNamingSymbol().name());
    if (s->isGlobal())
      llvm::errs() << ", global";
    if (s->initialValueSymbol)
      llvm::errs() << ", initial value: {" << *s->initialValueSymbol << "}";
  } else {
    llvm_unreachable("not a Variable");
  }
  llvm::errs() << '\n';
}

void Fortran::lower::pft::FunctionLikeUnit::dump() const {
  PFTDumper{}.dumpFunctionLikeUnit(llvm::errs(), *this);
}

void Fortran::lower::pft::ModuleLikeUnit::dump() const {
  PFTDumper{}.dumpModuleLikeUnit(llvm::errs(), *this);
}

/// The BlockDataUnit dump is just the associated symbol table.
void Fortran::lower::pft::BlockDataUnit::dump() const {
  llvm::errs() << "block data {\n" << symTab << "\n}\n";
}

std::vector<Fortran::lower::pft::Variable>
Fortran::lower::pft::buildFuncResultDependencyList(
    const Fortran::semantics::Symbol &symbol) {
  std::vector<std::vector<pft::Variable>> variableList;
  SymbolDependenceDepth sdd(variableList);
  sdd.analyzeAliasesInCurrentScope(symbol.owner());
  sdd.analyze(symbol);
  sdd.finalize();
  // Remove the pft::variable for the result itself, only its dependencies
  // should be returned in the list.
  assert(!variableList[0].empty() && "must at least contain the result");
  assert(&variableList[0].back().getSymbol() == &symbol &&
         "result sym should be last");
  variableList[0].pop_back();
  return variableList[0];
}

namespace {
/// Helper class to find all the symbols referenced in a FunctionLikeUnit.
/// It defines a parse tree visitor doing a deep visit in all nodes with
/// symbols (including evaluate::Expr).
struct SymbolVisitor {
  template <typename A>
  bool Pre(const A &x) {
    if constexpr (Fortran::parser::HasTypedExpr<A>::value)
      // Some parse tree Expr may legitimately be un-analyzed after semantics
      // (for instance PDT component initial value in the PDT definition body).
      if (const auto *expr = Fortran::semantics::GetExpr(nullptr, x))
        visitExpr(*expr);
    return true;
  }

  bool Pre(const Fortran::parser::Name &name) {
    if (const semantics::Symbol *symbol = name.symbol)
      visitSymbol(*symbol);
    return false;
  }

  template <typename T>
  void visitExpr(const Fortran::evaluate::Expr<T> &expr) {
    for (const semantics::Symbol &symbol :
         Fortran::evaluate::CollectSymbols(expr))
      visitSymbol(symbol);
  }

  void visitSymbol(const Fortran::semantics::Symbol &symbol) {
    callBack(symbol);
    // - Visit statement function body since it will be inlined in lowering.
    // - Visit function results specification expressions because allocations
    //   happens on the caller side.
    if (const auto *subprogramDetails =
            symbol.detailsIf<Fortran::semantics::SubprogramDetails>()) {
      if (const auto &maybeExpr = subprogramDetails->stmtFunction()) {
        visitExpr(*maybeExpr);
      } else {
        if (subprogramDetails->isFunction()) {
          // Visit result extents expressions that are explicit.
          const Fortran::semantics::Symbol &result =
              subprogramDetails->result();
          if (const auto *objectDetails =
                  result.detailsIf<Fortran::semantics::ObjectEntityDetails>())
            if (objectDetails->shape().IsExplicitShape())
              for (const Fortran::semantics::ShapeSpec &shapeSpec :
                   objectDetails->shape()) {
                visitExpr(shapeSpec.lbound().GetExplicit().value());
                visitExpr(shapeSpec.ubound().GetExplicit().value());
              }
        }
      }
    }
    if (Fortran::semantics::IsProcedure(symbol)) {
      if (auto dynamicType = Fortran::evaluate::DynamicType::From(symbol)) {
        // Visit result length specification expressions that are explicit.
        if (dynamicType->category() ==
            Fortran::common::TypeCategory::Character) {
          if (std::optional<Fortran::evaluate::ExtentExpr> length =
                  dynamicType->GetCharLength())
            visitExpr(*length);
        } else if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
                       Fortran::evaluate::GetDerivedTypeSpec(dynamicType)) {
          for (const auto &[_, param] : derivedTypeSpec->parameters())
            if (const Fortran::semantics::MaybeIntExpr &expr =
                    param.GetExplicit())
              visitExpr(expr.value());
        }
      }
    }
  }

  template <typename A>
  constexpr void Post(const A &) {}

  const std::function<void(const Fortran::semantics::Symbol &)> &callBack;
};
} // namespace

void Fortran::lower::pft::visitAllSymbols(
    const Fortran::lower::pft::FunctionLikeUnit &funit,
    const std::function<void(const Fortran::semantics::Symbol &)> callBack) {
  SymbolVisitor visitor{callBack};
  funit.visit([&](const auto &functionParserNode) {
    parser::Walk(functionParserNode, visitor);
  });
}

void Fortran::lower::pft::visitAllSymbols(
    const Fortran::lower::pft::Evaluation &eval,
    const std::function<void(const Fortran::semantics::Symbol &)> callBack) {
  SymbolVisitor visitor{callBack};
  eval.visit([&](const auto &functionParserNode) {
    parser::Walk(functionParserNode, visitor);
  });
}
