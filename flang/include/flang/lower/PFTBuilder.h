//===-- include/flang/lower/PFTBuilder.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_PFT_BUILDER_H_
#define FORTRAN_LOWER_PFT_BUILDER_H_

#include "flang/common/template.h"
#include "flang/parser/parse-tree.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

/// Build a light-weight tree over the parse-tree to help with lowering to FIR.
/// It is named Pre-FIR Tree (PFT) to underline it has no other usage than
/// helping lowering to FIR.
/// The PFT will capture pointers back into the parse tree, so the parse tree
/// data structure may <em>not</em> be changed between the construction of the
/// PFT and all of its uses.
///
/// The PFT captures a structured view of the program.  The program is a list of
/// units.  Function like units will contain lists of evaluations.  Evaluations
/// are either statements or constructs, where a construct contains a list of
/// evaluations. The resulting PFT structure can then be used to create FIR.

namespace Fortran::lower {
namespace pft {

struct Evaluation;
struct Program;
struct ModuleLikeUnit;
struct FunctionLikeUnit;

// TODO: A collection of Evaluations can obviously be any of the container
// types; leaving this as a std::list _for now_ because we reserve the right to
// insert PFT nodes in any order in O(1) time.
using EvaluationCollection = std::list<Evaluation>;

struct ParentType {
  template <typename A>
  ParentType(A &parent) : p{&parent} {}
  const std::variant<Program *, ModuleLikeUnit *, FunctionLikeUnit *,
                     Evaluation *>
      p;
};

/// Flags to describe the impact of parse-trees nodes on the program
/// control flow. These annotations to parse-tree nodes are later used to
/// build the control flow graph when lowering to FIR.
enum class CFGAnnotation {
  None,            // Node does not impact control flow.
  Goto,            // Node acts like a goto on the control flow.
  CondGoto,        // Node acts like a conditional goto on the control flow.
  IndGoto,         // Node acts like an indirect goto on the control flow.
  IoSwitch,        // Node is an IO statement with ERR, END, or EOR specifier.
  Switch,          // Node acts like a switch on the control flow.
  Iterative,       // Node creates iterations in the control flow.
  FirStructuredOp, // Node is a structured loop.
  Return,          // Node triggers a return from the current procedure.
  Terminate        // Node terminates the program.
};

/// Compiler-generated jump
///
/// This is used to convert implicit control-flow edges to explicit form in the
/// decorated PFT
struct CGJump {
  CGJump(Evaluation &to) : target{to} {}
  Evaluation &target;
};

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

using Constructs =
    std::tuple<parser::AssociateConstruct, parser::BlockConstruct,
               parser::CaseConstruct, parser::ChangeTeamConstruct,
               parser::CriticalConstruct, parser::DoConstruct,
               parser::IfConstruct, parser::SelectRankConstruct,
               parser::SelectTypeConstruct, parser::WhereConstruct,
               parser::ForallConstruct, parser::CompilerDirective,
               parser::OpenMPConstruct, parser::OmpEndLoopDirective>;

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

template <typename A>
constexpr static bool isActionStmt{common::HasMember<A, ActionStmts>};

template <typename A>
constexpr static bool isConstruct{common::HasMember<A, Constructs>};

template <typename A>
constexpr static bool isConstructStmt{common::HasMember<A, ConstructStmts>};

template <typename A>
constexpr static bool isOtherStmt{common::HasMember<A, OtherStmts>};

template <typename A>
constexpr static bool isGenerated{std::is_same_v<A, CGJump>};

template <typename A>
constexpr static bool isFunctionLike{common::HasMember<
    A, std::tuple<parser::MainProgram, parser::FunctionSubprogram,
                  parser::SubroutineSubprogram,
                  parser::SeparateModuleSubprogram>>};

/// Function-like units can contains lists of evaluations.  These can be
/// (simple) statements or constructs, where a construct contains its own
/// evaluations.
struct Evaluation {
  using EvalTuple = common::CombineTuples<ActionStmts, OtherStmts, Constructs,
                                          ConstructStmts>;

  /// Hide non-nullable pointers to the parse-tree node.
  template <typename A>
  using MakeRefType = const A *const;
  using EvalVariant =
      common::CombineVariants<common::MapTemplate<MakeRefType, EvalTuple>,
                              std::variant<CGJump>>;
  template <typename A>
  constexpr auto visit(A visitor) const {
    return std::visit(common::visitors{
                          [&](const auto *p) { return visitor(*p); },
                          [&](auto &r) { return visitor(r); },
                      },
                      u);
  }
  template <typename A>
  constexpr const A *getIf() const {
    if constexpr (!std::is_same_v<A, CGJump>) {
      if (auto *ptr{std::get_if<MakeRefType<A>>(&u)}) {
        return *ptr;
      }
    } else {
      return std::get_if<CGJump>(&u);
    }
    return nullptr;
  }
  template <typename A>
  constexpr bool isA() const {
    if constexpr (!std::is_same_v<A, CGJump>) {
      return std::holds_alternative<MakeRefType<A>>(u);
    }
    return std::holds_alternative<CGJump>(u);
  }

  Evaluation() = delete;
  Evaluation(const Evaluation &) = delete;
  Evaluation(Evaluation &&) = default;

  /// General ctor
  template <typename A>
  Evaluation(const A &a, const ParentType &p, const parser::CharBlock &pos,
             const std::optional<parser::Label> &lab)
      : u{&a}, parent{p}, pos{pos}, lab{lab} {}

  /// Compiler-generated jump
  Evaluation(const CGJump &jump, const ParentType &p)
      : u{jump}, parent{p}, cfg{CFGAnnotation::Goto} {}

  /// Construct ctor
  template <typename A>
  Evaluation(const A &a, const ParentType &parent) : u{&a}, parent{parent} {
    static_assert(pft::isConstruct<A>, "must be a construct");
  }

  constexpr bool isActionOrGenerated() const {
    return visit(common::visitors{
        [](auto &r) {
          using T = std::decay_t<decltype(r)>;
          return isActionStmt<T> || isGenerated<T>;
        },
    });
  }

  constexpr bool isStmt() const {
    return visit(common::visitors{
        [](auto &r) {
          using T = std::decay_t<decltype(r)>;
          static constexpr bool isStmt{isActionStmt<T> || isOtherStmt<T> ||
                                       isConstructStmt<T>};
          static_assert(!(isStmt && pft::isConstruct<T>),
                        "statement classification is inconsistent");
          return isStmt;
        },
    });
  }
  constexpr bool isConstruct() const { return !isStmt(); }

  /// Set the type of originating control flow type for this evaluation.
  void setCFG(CFGAnnotation a, Evaluation *cstr) {
    cfg = a;
    setBranches(cstr);
  }

  /// Is this evaluation a control-flow origin? (The PFT must be annotated)
  bool isControlOrigin() const { return cfg != CFGAnnotation::None; }

  /// Is this evaluation a control-flow target? (The PFT must be annotated)
  bool isControlTarget() const { return isTarget; }

  /// Set the containsBranches flag iff this evaluation (a construct) contains
  /// control flow
  void setBranches() { containsBranches = true; }

  EvaluationCollection *getConstructEvals() {
    auto *evals{subs.get()};
    if (isStmt() && !evals) {
      return nullptr;
    }
    if (isConstruct() && evals) {
      return evals;
    }
    llvm_unreachable("evaluation subs is inconsistent");
    return nullptr;
  }

  /// Set that the construct `cstr` (if not a nullptr) has branches.
  static void setBranches(Evaluation *cstr) {
    if (cstr)
      cstr->setBranches();
  }

  EvalVariant u;
  ParentType parent;
  parser::CharBlock pos;
  std::optional<parser::Label> lab;
  std::unique_ptr<EvaluationCollection> subs; // construct sub-statements
  CFGAnnotation cfg{CFGAnnotation::None};
  bool isTarget{false};         // this evaluation is a control target
  bool containsBranches{false}; // construct contains branches
};

/// A program is a list of program units.
/// These units can be function like, module like, or block data
struct ProgramUnit {
  template <typename A>
  ProgramUnit(const A &ptr, const ParentType &parent)
      : p{&ptr}, parent{parent} {}
  ProgramUnit(ProgramUnit &&) = default;
  ProgramUnit(const ProgramUnit &) = delete;

  const std::variant<
      const parser::MainProgram *, const parser::FunctionSubprogram *,
      const parser::SubroutineSubprogram *, const parser::Module *,
      const parser::Submodule *, const parser::SeparateModuleSubprogram *,
      const parser::BlockData *>
      p;
  ParentType parent;
};

/// Function-like units have similar structure. They all can contain executable
/// statements as well as other function-like units (internal procedures and
/// function statements).
struct FunctionLikeUnit : public ProgramUnit {
  // wrapper statements for function-like syntactic structures
  using FunctionStatement =
      std::variant<const parser::Statement<parser::ProgramStmt> *,
                   const parser::Statement<parser::EndProgramStmt> *,
                   const parser::Statement<parser::FunctionStmt> *,
                   const parser::Statement<parser::EndFunctionStmt> *,
                   const parser::Statement<parser::SubroutineStmt> *,
                   const parser::Statement<parser::EndSubroutineStmt> *,
                   const parser::Statement<parser::MpSubprogramStmt> *,
                   const parser::Statement<parser::EndMpSubprogramStmt> *>;

  FunctionLikeUnit(const parser::MainProgram &f, const ParentType &parent);
  FunctionLikeUnit(const parser::FunctionSubprogram &f,
                   const ParentType &parent);
  FunctionLikeUnit(const parser::SubroutineSubprogram &f,
                   const ParentType &parent);
  FunctionLikeUnit(const parser::SeparateModuleSubprogram &f,
                   const ParentType &parent);
  FunctionLikeUnit(FunctionLikeUnit &&) = default;
  FunctionLikeUnit(const FunctionLikeUnit &) = delete;

  bool isMainProgram() {
    return std::holds_alternative<
        const parser::Statement<parser::EndProgramStmt> *>(endStmt);
  }
  const parser::FunctionStmt *getFunction() {
    return getA<parser::FunctionStmt>();
  }
  const parser::SubroutineStmt *getSubroutine() {
    return getA<parser::SubroutineStmt>();
  }
  const parser::MpSubprogramStmt *getMPSubp() {
    return getA<parser::MpSubprogramStmt>();
  }

  /// Anonymous programs do not have a begin statement
  std::optional<FunctionStatement> beginStmt;
  FunctionStatement endStmt;
  EvaluationCollection evals;        // statements
  std::list<FunctionLikeUnit> funcs; // internal procedures

private:
  template <typename A>
  const A *getA() {
    if (beginStmt) {
      if (auto p =
              std::get_if<const parser::Statement<A> *>(&beginStmt.value()))
        return &(*p)->statement;
    }
    return nullptr;
  }
};

/// Module-like units have similar structure. They all can contain a list of
/// function-like units.
struct ModuleLikeUnit : public ProgramUnit {
  // wrapper statements for module-like syntactic structures
  using ModuleStatement =
      std::variant<const parser::Statement<parser::ModuleStmt> *,
                   const parser::Statement<parser::EndModuleStmt> *,
                   const parser::Statement<parser::SubmoduleStmt> *,
                   const parser::Statement<parser::EndSubmoduleStmt> *>;

  ModuleLikeUnit(const parser::Module &m, const ParentType &parent);
  ModuleLikeUnit(const parser::Submodule &m, const ParentType &parent);
  ~ModuleLikeUnit() = default;
  ModuleLikeUnit(ModuleLikeUnit &&) = default;
  ModuleLikeUnit(const ModuleLikeUnit &) = delete;

  ModuleStatement beginStmt;
  ModuleStatement endStmt;
  std::list<FunctionLikeUnit> funcs;
};

struct BlockDataUnit : public ProgramUnit {
  BlockDataUnit(const parser::BlockData &bd, const ParentType &parent);
  BlockDataUnit(BlockDataUnit &&) = default;
  BlockDataUnit(const BlockDataUnit &) = delete;
};

/// A Program is the top-level PFT
struct Program {
  using Units = std::variant<FunctionLikeUnit, ModuleLikeUnit, BlockDataUnit>;

  Program() = default;
  Program(Program &&) = default;
  Program(const Program &) = delete;

  std::list<Units> &getUnits() { return units; }

private:
  std::list<Units> units;
};

} // namespace pft

/// Create an PFT from the parse tree
std::unique_ptr<pft::Program> createPFT(const parser::Program &root);

/// Decorate the PFT with control flow annotations
///
/// The PFT must be decorated with control-flow annotations to prepare it for
/// use in generating a CFG-like structure.
void annotateControl(pft::Program &);

void dumpPFT(llvm::raw_ostream &o, pft::Program &);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_PFT_BUILDER_H_
