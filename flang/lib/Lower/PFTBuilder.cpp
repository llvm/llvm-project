//===-- lib/Lower/PFTBuilder.cc -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cassert>
#include <utility>

namespace Fortran::lower {
namespace {

/// Helpers to unveil parser node inside parser::Statement<>,
/// parser::UnlabeledStatement, and common::Indirection<>
template <typename A>
struct RemoveIndirectionHelper {
  using Type = A;
  static constexpr const Type &unwrap(const A &a) { return a; }
};
template <typename A>
struct RemoveIndirectionHelper<common::Indirection<A>> {
  using Type = A;
  static constexpr const Type &unwrap(const common::Indirection<A> &a) {
    return a.value();
  }
};

template <typename A>
const auto &removeIndirection(const A &a) {
  return RemoveIndirectionHelper<A>::unwrap(a);
}

template <typename A>
struct UnwrapStmt {
  static constexpr bool isStmt{false};
};
template <typename A>
struct UnwrapStmt<parser::Statement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::Statement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, pos{a.source}, lab{a.label} {
  }
  const Type &unwrapped;
  parser::CharBlock pos;
  std::optional<parser::Label> lab;
};
template <typename A>
struct UnwrapStmt<parser::UnlabeledStatement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::UnlabeledStatement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, pos{a.source} {}
  const Type &unwrapped;
  parser::CharBlock pos;
  std::optional<parser::Label> lab;
};

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time, so one goal here is to limit
/// the bridge to one such instantiation.
class PFTBuilder {
public:
  PFTBuilder() : pgm{new pft::Program}, parents{*pgm.get()} {}

  /// Get the result
  std::unique_ptr<pft::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    bool visit{true};
    if constexpr (pft::isFunctionLike<A>) {
      return enterFunc(a);
    } else if constexpr (pft::isConstruct<A>) {
      return enterConstruct(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (pft::isConstructStmt<T> || pft::isOtherStmt<T>) {
        addEval(pft::Evaluation{stmt.unwrapped, parents.back(), stmt.pos,
                                stmt.lab});
        visit = false;
      } else if constexpr (std::is_same_v<T, parser::ActionStmt>) {
        addEval(makeEvalAction(stmt.unwrapped, stmt.pos, stmt.lab));
        visit = false;
      }
    }
    return visit;
  }

  template <typename A>
  constexpr void Post(const A &) {
    if constexpr (pft::isFunctionLike<A>) {
      exitFunc();
    } else if constexpr (pft::isConstruct<A>) {
      exitConstruct();
    }
  }

  // Module like
  bool Pre(const parser::Module &node) { return enterModule(node); }
  bool Pre(const parser::Submodule &node) { return enterModule(node); }

  void Post(const parser::Module &) { exitModule(); }
  void Post(const parser::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const parser::BlockData &node) {
    addUnit(pft::BlockDataUnit{node, parents.back()});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const parser::UnlabeledStatement<parser::ForallAssignmentStmt>
               &statement) {
    addEval(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{x, parents.back(), statement.source, {}};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::Statement<parser::ForallAssignmentStmt> &statement) {
    addEval(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{x, parents.back(), statement.source,
                                 statement.label};
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
              addEval(pft::Evaluation{stmt.statement, parents.back(),
                                      stmt.source, stmt.label});
              return false;
            },
            [&](const auto &) { return true; },
        },
        whereBody.u);
  }

private:
  // ActionStmt has a couple of non-conforming cases, which get handled
  // explicitly here.  The other cases use an Indirection, which we discard in
  // the PFT.
  pft::Evaluation makeEvalAction(const parser::ActionStmt &statement,
                                 parser::CharBlock pos,
                                 std::optional<parser::Label> lab) {
    return std::visit(
        common::visitors{
            [&](const auto &x) {
              return pft::Evaluation{removeIndirection(x), parents.back(), pos,
                                     lab};
            },
        },
        statement.u);
  }

  // When we enter a function-like structure, we want to build a new unit and
  // set the builder's cursors to point to it.
  template <typename A>
  bool enterFunc(const A &func) {
    auto &unit = addFunc(pft::FunctionLikeUnit{func, parents.back()});
    funclist = &unit.funcs;
    pushEval(&unit.evals);
    parents.emplace_back(unit);
    return true;
  }
  /// Make funclist to point to current parent function list if it exists.
  void setFunctListToParentFuncs() {
    if (!parents.empty()) {
      std::visit(common::visitors{
                     [&](pft::FunctionLikeUnit *p) { funclist = &p->funcs; },
                     [&](pft::ModuleLikeUnit *p) { funclist = &p->funcs; },
                     [&](auto *) { funclist = nullptr; },
                 },
                 parents.back().p);
    }
  }

  void exitFunc() {
    popEval();
    parents.pop_back();
    setFunctListToParentFuncs();
  }

  // When we enter a construct structure, we want to build a new construct and
  // set the builder's evaluation cursor to point to it.
  template <typename A>
  bool enterConstruct(const A &construct) {
    auto &con = addEval(pft::Evaluation{construct, parents.back()});
    con.subs.reset(new pft::EvaluationCollection);
    pushEval(con.subs.get());
    parents.emplace_back(con);
    return true;
  }

  void exitConstruct() {
    popEval();
    parents.pop_back();
  }

  // When we enter a module structure, we want to build a new module and
  // set the builder's function cursor to point to it.
  template <typename A>
  bool enterModule(const A &func) {
    auto &unit = addUnit(pft::ModuleLikeUnit{func, parents.back()});
    funclist = &unit.funcs;
    parents.emplace_back(unit);
    return true;
  }

  void exitModule() {
    parents.pop_back();
    setFunctListToParentFuncs();
  }

  template <typename A>
  A &addUnit(A &&unit) {
    pgm->getUnits().emplace_back(std::move(unit));
    return std::get<A>(pgm->getUnits().back());
  }

  template <typename A>
  A &addFunc(A &&func) {
    if (funclist) {
      funclist->emplace_back(std::move(func));
      return funclist->back();
    }
    return addUnit(std::move(func));
  }

  /// move the Evaluation to the end of the current list
  pft::Evaluation &addEval(pft::Evaluation &&eval) {
    assert(funclist && "not in a function");
    assert(evallist.size() > 0);
    evallist.back()->emplace_back(std::move(eval));
    return evallist.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEval(pft::EvaluationCollection *eval) {
    assert(funclist && "not in a function");
    assert(eval && eval->empty() && "evaluation list isn't correct");
    evallist.emplace_back(eval);
  }

  /// pop the current list and return to the last Evaluation list
  void popEval() {
    assert(funclist && "not in a function");
    evallist.pop_back();
  }

  std::unique_ptr<pft::Program> pgm;
  /// funclist points to FunctionLikeUnit::funcs list (resp.
  /// ModuleLikeUnit::funcs) when building a FunctionLikeUnit (resp.
  /// ModuleLikeUnit) to store internal procedures (resp. module procedures).
  /// Otherwise (e.g. when building the top level Program), it is null.
  std::list<pft::FunctionLikeUnit> *funclist{nullptr};
  /// evallist is a stack of pointer to FunctionLikeUnit::evals (or
  /// Evaluation::subs) that are being build.
  std::vector<pft::EvaluationCollection *> evallist;
  std::vector<pft::ParentType> parents;
};

template <typename Label, typename A>
constexpr bool hasLabel(const A &stmt) {
  auto isLabel{
      [](const auto &v) { return std::holds_alternative<Label>(v.u); }};
  if constexpr (std::is_same_v<A, parser::ReadStmt> ||
                std::is_same_v<A, parser::WriteStmt>) {
    return std::any_of(std::begin(stmt.controls), std::end(stmt.controls),
                       isLabel);
  }
  if constexpr (std::is_same_v<A, parser::WaitStmt>) {
    return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
  }
  if constexpr (std::is_same_v<Label, parser::ErrLabel>) {
    if constexpr (common::HasMember<
                      A, std::tuple<parser::OpenStmt, parser::CloseStmt,
                                    parser::BackspaceStmt, parser::EndfileStmt,
                                    parser::RewindStmt, parser::FlushStmt>>)
      return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
    if constexpr (std::is_same_v<A, parser::InquireStmt>) {
      const auto &specifiers{std::get<std::list<parser::InquireSpec>>(stmt.u)};
      return std::any_of(std::begin(specifiers), std::end(specifiers), isLabel);
    }
  }
  return false;
}

bool hasAltReturns(const parser::CallStmt &callStmt) {
  const auto &args{std::get<std::list<parser::ActualArgSpec>>(callStmt.v.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<parser::ActualArg>(arg.t)};
    if (std::holds_alternative<parser::AltReturnSpec>(actual.u))
      return true;
  }
  return false;
}

/// Determine if `callStmt` has alternate returns and if so set `e` to be the
/// origin of a switch-like control flow
///
/// \param cstr points to the current construct. It may be null at the top-level
/// of a FunctionLikeUnit.
void altRet(pft::Evaluation &evaluation, const parser::CallStmt &callStmt,
            pft::Evaluation *cstr) {
  if (hasAltReturns(callStmt))
    evaluation.setCFG(pft::CFGAnnotation::Switch, cstr);
}

/// \param cstr points to the current construct. It may be null at the top-level
/// of a FunctionLikeUnit.
void annotateEvalListCFG(pft::EvaluationCollection &evaluationCollection,
                         pft::Evaluation *cstr) {
  bool nextIsTarget = false;
  for (auto &eval : evaluationCollection) {
    eval.isTarget = nextIsTarget;
    nextIsTarget = false;
    if (auto *subs{eval.getConstructEvals()}) {
      annotateEvalListCFG(*subs, &eval);
      // assume that the entry and exit are both possible branch targets
      nextIsTarget = true;
    }

    if (eval.isActionOrGenerated() && eval.lab.has_value())
      eval.isTarget = true;
    eval.visit(common::visitors{
        [&](const parser::CallStmt &statement) {
          altRet(eval, statement, cstr);
        },
        [&](const parser::CycleStmt &) {
          eval.setCFG(pft::CFGAnnotation::Goto, cstr);
        },
        [&](const parser::ExitStmt &) {
          eval.setCFG(pft::CFGAnnotation::Goto, cstr);
        },
        [&](const parser::FailImageStmt &) {
          eval.setCFG(pft::CFGAnnotation::Terminate, cstr);
        },
        [&](const parser::GotoStmt &) {
          eval.setCFG(pft::CFGAnnotation::Goto, cstr);
        },
        [&](const parser::IfStmt &) {
          eval.setCFG(pft::CFGAnnotation::CondGoto, cstr);
        },
        [&](const parser::ReturnStmt &) {
          eval.setCFG(pft::CFGAnnotation::Return, cstr);
        },
        [&](const parser::StopStmt &) {
          eval.setCFG(pft::CFGAnnotation::Terminate, cstr);
        },
        [&](const parser::ArithmeticIfStmt &) {
          eval.setCFG(pft::CFGAnnotation::Switch, cstr);
        },
        [&](const parser::AssignedGotoStmt &) {
          eval.setCFG(pft::CFGAnnotation::IndGoto, cstr);
        },
        [&](const parser::ComputedGotoStmt &) {
          eval.setCFG(pft::CFGAnnotation::Switch, cstr);
        },
        [&](const parser::WhereStmt &) {
          // fir.loop + fir.where around the next stmt
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Iterative, cstr);
        },
        [&](const parser::ForallStmt &) {
          // fir.loop around the next stmt
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Iterative, cstr);
        },
        [&](pft::CGJump &) { eval.setCFG(pft::CFGAnnotation::Goto, cstr); },
        [&](const parser::SelectCaseStmt &) {
          eval.setCFG(pft::CFGAnnotation::Switch, cstr);
        },
        [&](const parser::NonLabelDoStmt &) {
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Iterative, cstr);
        },
        [&](const parser::EndDoStmt &) {
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Goto, cstr);
        },
        [&](const parser::IfThenStmt &) {
          eval.setCFG(pft::CFGAnnotation::CondGoto, cstr);
        },
        [&](const parser::ElseIfStmt &) {
          eval.setCFG(pft::CFGAnnotation::CondGoto, cstr);
        },
        [&](const parser::SelectRankStmt &) {
          eval.setCFG(pft::CFGAnnotation::Switch, cstr);
        },
        [&](const parser::SelectTypeStmt &) {
          eval.setCFG(pft::CFGAnnotation::Switch, cstr);
        },
        [&](const parser::WhereConstruct &) {
          // mark the WHERE as if it were a DO loop
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Iterative, cstr);
        },
        [&](const parser::WhereConstructStmt &) {
          eval.setCFG(pft::CFGAnnotation::CondGoto, cstr);
        },
        [&](const parser::MaskedElsewhereStmt &) {
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::CondGoto, cstr);
        },
        [&](const parser::ForallConstructStmt &) {
          eval.isTarget = true;
          eval.setCFG(pft::CFGAnnotation::Iterative, cstr);
        },

        [&](const auto &stmt) {
          // Handle statements with similar impact on control flow
          using IoStmts = std::tuple<parser::BackspaceStmt, parser::CloseStmt,
                                     parser::EndfileStmt, parser::FlushStmt,
                                     parser::InquireStmt, parser::OpenStmt,
                                     parser::ReadStmt, parser::RewindStmt,
                                     parser::WaitStmt, parser::WriteStmt>;

          using TargetStmts =
              std::tuple<parser::EndAssociateStmt, parser::EndBlockStmt,
                         parser::CaseStmt, parser::EndSelectStmt,
                         parser::EndChangeTeamStmt, parser::EndCriticalStmt,
                         parser::ElseStmt, parser::EndIfStmt,
                         parser::SelectRankCaseStmt, parser::TypeGuardStmt,
                         parser::ElsewhereStmt, parser::EndWhereStmt,
                         parser::EndForallStmt>;

          using DoNothingConstructStmts =
              std::tuple<parser::BlockStmt, parser::AssociateStmt,
                         parser::CriticalStmt, parser::ChangeTeamStmt>;

          using A = std::decay_t<decltype(stmt)>;
          if constexpr (common::HasMember<A, IoStmts>) {
            if (hasLabel<parser::ErrLabel>(stmt) ||
                hasLabel<parser::EorLabel>(stmt) ||
                hasLabel<parser::EndLabel>(stmt))
              eval.setCFG(pft::CFGAnnotation::IoSwitch, cstr);
          } else if constexpr (common::HasMember<A, TargetStmts>) {
            eval.isTarget = true;
          } else if constexpr (common::HasMember<A, DoNothingConstructStmts>) {
            // Explicitly do nothing for these construct statements
          } else {
            static_assert(!pft::isConstructStmt<A>,
                          "All ConstructStmts impact on the control flow "
                          "should be explicitly handled");
          }
          /* else do nothing */
        },
    });
  }
}

/// Annotate the PFT with CFG source decorations (see CFGAnnotation) and mark
/// potential branch targets
inline void annotateFuncCFG(pft::FunctionLikeUnit &functionLikeUnit) {
  annotateEvalListCFG(functionLikeUnit.evals, nullptr);
  for (auto &internalFunc : functionLikeUnit.funcs)
    annotateFuncCFG(internalFunc);
}

class PFTDumper {
public:
  void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(common::visitors{
                     [&](pft::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEndBlockData\n\n";
                     },
                     [&](pft::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](pft::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
    resetIndexes();
  }

  llvm::StringRef evalName(pft::Evaluation &eval) {
    return eval.visit(common::visitors{
        [](const pft::CGJump) { return "CGJump"; },
        [](const auto &parseTreeNode) {
          return parser::ParseTreeDumper::GetNodeName(parseTreeNode);
        },
    });
  }

  void dumpEvalList(llvm::raw_ostream &outputStream,
                    pft::EvaluationCollection &evaluationCollection,
                    int indent = 1) {
    static const std::string white{"                                      ++"};
    std::string indentString{white.substr(0, indent * 2)};
    for (pft::Evaluation &eval : evaluationCollection) {
      outputStream << indentString << getNodeIndex(eval) << " ";
      llvm::StringRef name{evalName(eval)};
      if (auto *subs{eval.getConstructEvals()}) {
        outputStream << "<<" << name << ">>";
        outputStream << "\n";
        dumpEvalList(outputStream, *subs, indent + 1);
        outputStream << indentString << "<<End" << name << ">>\n";
      } else {
        outputStream << name;
        outputStream << ": " << eval.pos.ToString() + "\n";
      }
    }
  }

  void dumpFunctionLikeUnit(llvm::raw_ostream &outputStream,
                            pft::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    llvm::StringRef unitKind{};
    std::string name{};
    std::string header{};
    if (functionLikeUnit.beginStmt) {
      std::visit(
          common::visitors{
              [&](const parser::Statement<parser::ProgramStmt> *statement) {
                unitKind = "Program";
                name = statement->statement.v.ToString();
              },
              [&](const parser::Statement<parser::FunctionStmt> *statement) {
                unitKind = "Function";
                name =
                    std::get<parser::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const parser::Statement<parser::SubroutineStmt> *statement) {
                unitKind = "Subroutine";
                name =
                    std::get<parser::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const parser::Statement<parser::MpSubprogramStmt>
                      *statement) {
                unitKind = "MpSubprogram";
                name = statement->statement.v.ToString();
                header = statement->source.ToString();
              },
              [&](auto *) {},
          },
          *functionLikeUnit.beginStmt);
    } else {
      unitKind = "Program";
      name = "<anonymous>";
    }
    outputStream << unitKind << ' ' << name;
    if (header.size())
      outputStream << ": " << header;
    outputStream << '\n';
    dumpEvalList(outputStream, functionLikeUnit.evals);
    if (!functionLikeUnit.funcs.empty()) {
      outputStream << "\nContains\n";
      for (auto &func : functionLikeUnit.funcs)
        dumpFunctionLikeUnit(outputStream, func);
      outputStream << "EndContains\n";
    }
    outputStream << "End" << unitKind << ' ' << name << "\n\n";
  }

  void dumpModuleLikeUnit(llvm::raw_ostream &outputStream,
                          pft::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike: ";
    outputStream << "\nContains\n";
    for (auto &func : moduleLikeUnit.funcs)
      dumpFunctionLikeUnit(outputStream, func);
    outputStream << "EndContains\nEndModuleLike\n\n";
  }

  template <typename T>
  std::size_t getNodeIndex(const T &node) {
    auto addr{static_cast<const void *>(&node)};
    auto it{nodeIndexes.find(addr)};
    if (it != nodeIndexes.end()) {
      return it->second;
    }
    nodeIndexes.try_emplace(addr, nextIndex);
    return nextIndex++;
  }
  std::size_t getNodeIndex(const pft::Program &) { return 0; }

  void resetIndexes() {
    nodeIndexes.clear();
    nextIndex = 1;
  }

private:
  llvm::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

template <typename A, typename T>
pft::FunctionLikeUnit::FunctionStatement getFunctionStmt(const T &func) {
  return pft::FunctionLikeUnit::FunctionStatement{
      &std::get<parser::Statement<A>>(func.t)};
}
template <typename A, typename T>
pft::ModuleLikeUnit::ModuleStatement getModuleStmt(const T &mod) {
  return pft::ModuleLikeUnit::ModuleStatement{
      &std::get<parser::Statement<A>>(mod.t)};
}

} // namespace

pft::FunctionLikeUnit::FunctionLikeUnit(const parser::MainProgram &func,
                                        const pft::ParentType &parent)
    : ProgramUnit{func, parent} {
  auto &ps{
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(func.t)};
  if (ps.has_value()) {
    const parser::Statement<parser::ProgramStmt> &statement{ps.value()};
    beginStmt = &statement;
  }
  endStmt = getFunctionStmt<parser::EndProgramStmt>(func);
}

pft::FunctionLikeUnit::FunctionLikeUnit(const parser::FunctionSubprogram &func,
                                        const pft::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::FunctionStmt>(func)},
      endStmt{getFunctionStmt<parser::EndFunctionStmt>(func)} {}

pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SubroutineSubprogram &func, const pft::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<parser::EndSubroutineStmt>(func)} {}

pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SeparateModuleSubprogram &func, const pft::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<parser::EndMpSubprogramStmt>(func)} {}

pft::ModuleLikeUnit::ModuleLikeUnit(const parser::Module &m,
                                    const pft::ParentType &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::ModuleStmt>(m)},
      endStmt{getModuleStmt<parser::EndModuleStmt>(m)} {}

pft::ModuleLikeUnit::ModuleLikeUnit(const parser::Submodule &m,
                                    const pft::ParentType &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::SubmoduleStmt>(
                                  m)},
      endStmt{getModuleStmt<parser::EndSubmoduleStmt>(m)} {}

pft::BlockDataUnit::BlockDataUnit(const parser::BlockData &bd,
                                  const pft::ParentType &parent)
    : ProgramUnit{bd, parent} {}

std::unique_ptr<pft::Program> createPFT(const parser::Program &root) {
  PFTBuilder walker;
  Walk(root, walker);
  return walker.result();
}

void annotateControl(pft::Program &pft) {
  for (auto &unit : pft.getUnits()) {
    std::visit(common::visitors{
                   [](pft::BlockDataUnit &) {},
                   [](pft::FunctionLikeUnit &func) { annotateFuncCFG(func); },
                   [](pft::ModuleLikeUnit &unit) {
                     for (auto &func : unit.funcs)
                       annotateFuncCFG(func);
                   },
               },
               unit);
  }
}

/// Dump a PFT.
void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

} // namespace Fortran::lower
