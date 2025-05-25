//===-- lib/Semantics/resolve-labels.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "resolve-labels.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/template.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/semantics.h"
#include <cstdarg>
#include <type_traits>

namespace Fortran::semantics {

using namespace parser::literals;

ENUM_CLASS(
    TargetStatementEnum, Do, Branch, Format, CompatibleDo, CompatibleBranch)
using LabeledStmtClassificationSet =
    common::EnumSet<TargetStatementEnum, TargetStatementEnum_enumSize>;

using IndexList = std::vector<std::pair<parser::CharBlock, parser::CharBlock>>;
// A ProxyForScope is an integral proxy for a Fortran scope. This is required
// because the parse tree does not actually have the scopes required.
using ProxyForScope = unsigned;
// Minimal scope information
struct ScopeInfo {
  ProxyForScope parent{};
  bool isExteriorGotoFatal{false};
  int depth{0};
};
struct LabeledStatementInfoTuplePOD {
  ProxyForScope proxyForScope;
  parser::CharBlock parserCharBlock;
  LabeledStmtClassificationSet labeledStmtClassificationSet;
  bool isExecutableConstructEndStmt;
};
using TargetStmtMap = std::map<parser::Label, LabeledStatementInfoTuplePOD>;
struct SourceStatementInfoTuplePOD {
  SourceStatementInfoTuplePOD(const parser::Label &parserLabel,
      const ProxyForScope &proxyForScope,
      const parser::CharBlock &parserCharBlock)
      : parserLabel{parserLabel}, proxyForScope{proxyForScope},
        parserCharBlock{parserCharBlock} {}
  parser::Label parserLabel;
  ProxyForScope proxyForScope;
  parser::CharBlock parserCharBlock;
};
using SourceStmtList = std::vector<SourceStatementInfoTuplePOD>;
enum class Legality { never, always, formerly };

bool HasScope(ProxyForScope scope) { return scope != ProxyForScope{0u}; }

// F18:R1131
template <typename A>
constexpr Legality IsLegalDoTerm(const parser::Statement<A> &) {
  if (std::is_same_v<A, common::Indirection<parser::EndDoStmt>> ||
      std::is_same_v<A, parser::EndDoStmt>) {
    return Legality::always;
  } else if (std::is_same_v<A, parser::EndForallStmt> ||
      std::is_same_v<A, parser::EndWhereStmt>) {
    // Executable construct end statements are also supported as
    // an extension but they need special care because the associated
    // construct create their own scope.
    return Legality::formerly;
  } else {
    return Legality::never;
  }
}

constexpr Legality IsLegalDoTerm(
    const parser::Statement<parser::ActionStmt> &actionStmt) {
  if (std::holds_alternative<parser::ContinueStmt>(actionStmt.statement.u)) {
    // See F08:C816
    return Legality::always;
  } else if (!(std::holds_alternative<
                   common::Indirection<parser::ArithmeticIfStmt>>(
                   actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::CycleStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::ExitStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::StopStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::GotoStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<
                     common::Indirection<parser::ReturnStmt>>(
                     actionStmt.statement.u))) {
    return Legality::formerly;
  } else {
    return Legality::never;
  }
}

template <typename A> constexpr bool IsFormat(const parser::Statement<A> &) {
  return std::is_same_v<A, common::Indirection<parser::FormatStmt>>;
}

template <typename A>
constexpr Legality IsLegalBranchTarget(const parser::Statement<A> &) {
  if (std::is_same_v<A, parser::ActionStmt> ||
      std::is_same_v<A, parser::AssociateStmt> ||
      std::is_same_v<A, parser::EndAssociateStmt> ||
      std::is_same_v<A, parser::IfThenStmt> ||
      std::is_same_v<A, parser::EndIfStmt> ||
      std::is_same_v<A, parser::SelectCaseStmt> ||
      std::is_same_v<A, parser::EndSelectStmt> ||
      std::is_same_v<A, parser::SelectRankStmt> ||
      std::is_same_v<A, parser::SelectTypeStmt> ||
      std::is_same_v<A, common::Indirection<parser::LabelDoStmt>> ||
      std::is_same_v<A, parser::NonLabelDoStmt> ||
      std::is_same_v<A, parser::EndDoStmt> ||
      std::is_same_v<A, common::Indirection<parser::EndDoStmt>> ||
      std::is_same_v<A, parser::BlockStmt> ||
      std::is_same_v<A, parser::EndBlockStmt> ||
      std::is_same_v<A, parser::CriticalStmt> ||
      std::is_same_v<A, parser::EndCriticalStmt> ||
      std::is_same_v<A, parser::ForallConstructStmt> ||
      std::is_same_v<A, parser::WhereConstructStmt> ||
      std::is_same_v<A, parser::ChangeTeamStmt> ||
      std::is_same_v<A, parser::EndChangeTeamStmt> ||
      std::is_same_v<A, parser::EndFunctionStmt> ||
      std::is_same_v<A, parser::EndMpSubprogramStmt> ||
      std::is_same_v<A, parser::EndProgramStmt> ||
      std::is_same_v<A, parser::EndSubroutineStmt>) {
    return Legality::always;
  } else {
    return Legality::never;
  }
}

template <typename A>
constexpr LabeledStmtClassificationSet ConstructBranchTargetFlags(
    const parser::Statement<A> &statement) {
  LabeledStmtClassificationSet labeledStmtClassificationSet{};
  if (IsLegalDoTerm(statement) == Legality::always) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Do);
  } else if (IsLegalDoTerm(statement) == Legality::formerly) {
    labeledStmtClassificationSet.set(TargetStatementEnum::CompatibleDo);
  }
  if (IsLegalBranchTarget(statement) == Legality::always) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Branch);
  } else if (IsLegalBranchTarget(statement) == Legality::formerly) {
    labeledStmtClassificationSet.set(TargetStatementEnum::CompatibleBranch);
  }
  if (IsFormat(statement)) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Format);
  }
  return labeledStmtClassificationSet;
}

static unsigned SayLabel(parser::Label label) {
  return static_cast<unsigned>(label);
}

struct UnitAnalysis {
  UnitAnalysis() { scopeModel.emplace_back(); }

  SourceStmtList doStmtSources;
  SourceStmtList formatStmtSources;
  SourceStmtList otherStmtSources;
  SourceStmtList assignStmtSources;
  TargetStmtMap targetStmts;
  std::vector<ScopeInfo> scopeModel;
};

// Some parse tree record for statements simply wrap construct names;
// others include them as tuple components.  Given a statement,
// return a pointer to its name if it has one.
template <typename A>
const parser::CharBlock *GetStmtName(const parser::Statement<A> &stmt) {
  const std::optional<parser::Name> *name{nullptr};
  if constexpr (WrapperTrait<A>) {
    if constexpr (std::is_same_v<decltype(A::v), parser::Name>) {
      return &stmt.statement.v.source;
    } else {
      name = &stmt.statement.v;
    }
  } else if constexpr (std::is_same_v<A, parser::SelectRankStmt> ||
      std::is_same_v<A, parser::SelectTypeStmt>) {
    name = &std::get<0>(stmt.statement.t);
  } else if constexpr (common::HasMember<parser::Name,
                           decltype(stmt.statement.t)>) {
    return &std::get<parser::Name>(stmt.statement.t).source;
  } else {
    name = &std::get<std::optional<parser::Name>>(stmt.statement.t);
  }
  if (name && *name) {
    return &(*name)->source;
  }
  return nullptr;
}

class ParseTreeAnalyzer {
public:
  ParseTreeAnalyzer(ParseTreeAnalyzer &&that) = default;
  ParseTreeAnalyzer(SemanticsContext &context) : context_{context} {}

  template <typename A> constexpr bool Pre(const A &x) {
    using LabeledProgramUnitStmts =
        std::tuple<parser::MainProgram, parser::FunctionSubprogram,
            parser::SubroutineSubprogram, parser::SeparateModuleSubprogram>;
    if constexpr (common::HasMember<A, LabeledProgramUnitStmts>) {
      const auto &endStmt{std::get<std::tuple_size_v<decltype(x.t)> - 1>(x.t)};
      if (endStmt.label) {
        // The END statement for a subprogram appears after any internal
        // subprograms.  Visit that statement in advance so that results
        // are placed in the correct programUnits_ slot.
        auto targetFlags{ConstructBranchTargetFlags(endStmt)};
        AddTargetLabelDefinition(endStmt.label.value(), targetFlags,
            currentScope_,
            /*isExecutableConstructEndStmt=*/false);
      }
    }
    return true;
  }
  template <typename A> constexpr void Post(const A &) {}

  template <typename A> bool Pre(const parser::Statement<A> &statement) {
    currentPosition_ = statement.source;
    const auto &label = statement.label;
    if (!label) {
      return true;
    }
    using LabeledConstructStmts = std::tuple<parser::AssociateStmt,
        parser::BlockStmt, parser::ChangeTeamStmt, parser::CriticalStmt,
        parser::IfThenStmt, parser::NonLabelDoStmt, parser::SelectCaseStmt,
        parser::SelectRankStmt, parser::SelectTypeStmt,
        parser::ForallConstructStmt, parser::WhereConstructStmt>;
    using LabeledConstructEndStmts = std::tuple<parser::EndAssociateStmt,
        parser::EndBlockStmt, parser::EndChangeTeamStmt,
        parser::EndCriticalStmt, parser::EndDoStmt, parser::EndForallStmt,
        parser::EndIfStmt, parser::EndWhereStmt>;
    using LabeledProgramUnitEndStmts =
        std::tuple<parser::EndFunctionStmt, parser::EndMpSubprogramStmt,
            parser::EndProgramStmt, parser::EndSubroutineStmt>;
    auto targetFlags{ConstructBranchTargetFlags(statement)};
    if constexpr (common::HasMember<A, LabeledConstructStmts>) {
      AddTargetLabelDefinition(label.value(), targetFlags, ParentScope(),
          /*isExecutableConstructEndStmt=*/false);
    } else if constexpr (std::is_same_v<A, parser::EndIfStmt> ||
        std::is_same_v<A, parser::EndSelectStmt>) {
      // the label on an END IF/SELECT is not in the last part/case
      AddTargetLabelDefinition(label.value(), targetFlags, ParentScope(),
          /*isExecutableConstructEndStmt=*/true);
    } else if constexpr (common::HasMember<A, LabeledConstructEndStmts>) {
      AddTargetLabelDefinition(label.value(), targetFlags, currentScope_,
          /*isExecutableConstructEndStmt=*/true);
    } else if constexpr (!common::HasMember<A, LabeledProgramUnitEndStmts>) {
      // Program unit END statements have already been processed.
      AddTargetLabelDefinition(label.value(), targetFlags, currentScope_,
          /*isExecutableConstructEndStmt=*/false);
    }
    return true;
  }

  // see 11.1.1
  bool Pre(const parser::ProgramUnit &) { return InitializeNewScopeContext(); }
  bool Pre(const parser::InternalSubprogram &) {
    return InitializeNewScopeContext();
  }
  bool Pre(const parser::ModuleSubprogram &) {
    return InitializeNewScopeContext();
  }
  bool Pre(const parser::AssociateConstruct &associateConstruct) {
    return PushConstructName(associateConstruct);
  }
  bool Pre(const parser::BlockConstruct &blockConstruct) {
    return PushConstructName(blockConstruct);
  }
  bool Pre(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    return PushConstructName(changeTeamConstruct);
  }
  bool Pre(const parser::CriticalConstruct &criticalConstruct) {
    return PushConstructName(criticalConstruct);
  }
  bool Pre(const parser::DoConstruct &doConstruct) {
    const auto &optionalName{std::get<std::optional<parser::Name>>(
        std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)
            .statement.t)};
    if (optionalName) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    // Allow FORTRAN '66 extended DO ranges
    PushScope(false);
    // Process labels of the DO and END DO statements, but not the
    // statements themselves, so that a non-construct END DO
    // can be distinguished (below).
    Pre(std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t));
    Walk(std::get<parser::Block>(doConstruct.t), *this);
    Pre(std::get<parser::Statement<parser::EndDoStmt>>(doConstruct.t));
    PopConstructName(doConstruct);
    return false;
  }
  void Post(const parser::EndDoStmt &endDoStmt) {
    // Visited only for non-construct labeled DO termination
    if (const auto &name{endDoStmt.v}) {
      context_.Say(name->source, "Unexpected DO construct name '%s'"_err_en_US,
          name->source);
    }
  }
  bool Pre(const parser::IfConstruct &ifConstruct) {
    return PushConstructName(ifConstruct);
  }
  void Post(const parser::IfThenStmt &) { PushScope(false); }
  bool Pre(const parser::IfConstruct::ElseIfBlock &) {
    return SwitchToNewScope();
  }
  bool Pre(const parser::IfConstruct::ElseBlock &) {
    return SwitchToNewScope();
  }
  bool Pre(const parser::EndIfStmt &) {
    PopScope();
    return true;
  }
  bool Pre(const parser::CaseConstruct &caseConstruct) {
    return PushConstructName(caseConstruct);
  }
  void Post(const parser::SelectCaseStmt &) { PushScope(false); }
  bool Pre(const parser::CaseConstruct::Case &) { return SwitchToNewScope(); }
  bool Pre(const parser::SelectRankConstruct &selectRankConstruct) {
    return PushConstructName(selectRankConstruct);
  }
  void Post(const parser::SelectRankStmt &) { PushScope(true); }
  bool Pre(const parser::SelectRankConstruct::RankCase &) {
    return SwitchToNewScope();
  }
  bool Pre(const parser::SelectTypeConstruct &selectTypeConstruct) {
    return PushConstructName(selectTypeConstruct);
  }
  void Post(const parser::SelectTypeStmt &) { PushScope(true); }
  bool Pre(const parser::SelectTypeConstruct::TypeCase &) {
    return SwitchToNewScope();
  }
  void Post(const parser::EndSelectStmt &) { PopScope(); }
  bool Pre(const parser::WhereConstruct &whereConstruct) {
    return PushConstructName(whereConstruct);
  }
  bool Pre(const parser::ForallConstruct &forallConstruct) {
    return PushConstructName(forallConstruct);
  }

  void Post(const parser::AssociateConstruct &associateConstruct) {
    PopConstructName(associateConstruct);
  }
  void Post(const parser::BlockConstruct &blockConstruct) {
    PopConstructName(blockConstruct);
  }
  void Post(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    PopConstructName(changeTeamConstruct);
  }
  void Post(const parser::CriticalConstruct &criticalConstruct) {
    PopConstructName(criticalConstruct);
  }
  void Post(const parser::IfConstruct &ifConstruct) {
    PopConstructName(ifConstruct);
  }
  void Post(const parser::CaseConstruct &caseConstruct) {
    PopConstructName(caseConstruct);
  }
  void Post(const parser::SelectRankConstruct &selectRankConstruct) {
    PopConstructName(selectRankConstruct);
  }
  void Post(const parser::SelectTypeConstruct &selectTypeConstruct) {
    PopConstructName(selectTypeConstruct);
  }
  void Post(const parser::WhereConstruct &whereConstruct) {
    PopConstructName(whereConstruct);
  }
  void Post(const parser::ForallConstruct &forallConstruct) {
    PopConstructName(forallConstruct);
  }

  // Checks for missing or mismatching names on various constructs (e.g., IF)
  // and their intermediate or terminal statements that allow optional
  // construct names(e.g., ELSE).  When an optional construct name is present,
  // the construct as a whole must have a name that matches.
  template <typename FIRST, typename CONSTRUCT, typename STMT>
  void CheckOptionalName(const char *constructTag, const CONSTRUCT &a,
      const parser::Statement<STMT> &stmt) {
    if (const parser::CharBlock * name{GetStmtName(stmt)}) {
      const auto &firstStmt{std::get<parser::Statement<FIRST>>(a.t)};
      if (const parser::CharBlock * firstName{GetStmtName(firstStmt)}) {
        if (*firstName != *name) {
          context_.Say(*name, "%s name mismatch"_err_en_US, constructTag)
              .Attach(*firstName, "should be"_en_US);
        }
      } else {
        context_.Say(*name, "%s name not allowed"_err_en_US, constructTag)
            .Attach(firstStmt.source, "in unnamed %s"_en_US, constructTag);
      }
    }
  }

  // C1414
  void Post(const parser::BlockData &blockData) {
    CheckOptionalName<parser::BlockDataStmt>("BLOCK DATA subprogram", blockData,
        std::get<parser::Statement<parser::EndBlockDataStmt>>(blockData.t));
  }

  bool Pre(const parser::InterfaceBody &) {
    PushDisposableMap();
    return true;
  }
  void Post(const parser::InterfaceBody &) { PopDisposableMap(); }

  // C1564
  void Post(const parser::InterfaceBody::Function &func) {
    CheckOptionalName<parser::FunctionStmt>("FUNCTION", func,
        std::get<parser::Statement<parser::EndFunctionStmt>>(func.t));
  }

  // C1564
  void Post(const parser::FunctionSubprogram &functionSubprogram) {
    CheckOptionalName<parser::FunctionStmt>("FUNCTION", functionSubprogram,
        std::get<parser::Statement<parser::EndFunctionStmt>>(
            functionSubprogram.t));
  }

  // C1502
  void Post(const parser::InterfaceBlock &interfaceBlock) {
    if (const auto &endGenericSpec{
            std::get<parser::Statement<parser::EndInterfaceStmt>>(
                interfaceBlock.t)
                .statement.v}) {
      const auto &interfaceStmt{
          std::get<parser::Statement<parser::InterfaceStmt>>(interfaceBlock.t)};
      if (std::holds_alternative<parser::Abstract>(interfaceStmt.statement.u)) {
        context_
            .Say(endGenericSpec->source,
                "END INTERFACE generic name (%s) may not appear for ABSTRACT INTERFACE"_err_en_US,
                endGenericSpec->source)
            .Attach(
                interfaceStmt.source, "corresponding ABSTRACT INTERFACE"_en_US);
      } else if (const auto &genericSpec{
                     std::get<std::optional<parser::GenericSpec>>(
                         interfaceStmt.statement.u)}) {
        bool ok{genericSpec->source == endGenericSpec->source};
        if (!ok) {
          // Accept variant spellings of .LT. &c.
          const auto *endOp{
              std::get_if<parser::DefinedOperator>(&endGenericSpec->u)};
          const auto *op{std::get_if<parser::DefinedOperator>(&genericSpec->u)};
          if (endOp && op) {
            const auto *endIntrin{
                std::get_if<parser::DefinedOperator::IntrinsicOperator>(
                    &endOp->u)};
            const auto *intrin{
                std::get_if<parser::DefinedOperator::IntrinsicOperator>(
                    &op->u)};
            ok = endIntrin && intrin && *endIntrin == *intrin;
          }
        }
        if (!ok) {
          context_
              .Say(endGenericSpec->source,
                  "END INTERFACE generic name (%s) does not match generic INTERFACE (%s)"_err_en_US,
                  endGenericSpec->source, genericSpec->source)
              .Attach(genericSpec->source, "corresponding INTERFACE"_en_US);
        }
      } else {
        context_
            .Say(endGenericSpec->source,
                "END INTERFACE generic name (%s) may not appear for non-generic INTERFACE"_err_en_US,
                endGenericSpec->source)
            .Attach(interfaceStmt.source, "corresponding INTERFACE"_en_US);
      }
    }
  }

  // C1402
  void Post(const parser::Module &module) {
    CheckOptionalName<parser::ModuleStmt>("MODULE", module,
        std::get<parser::Statement<parser::EndModuleStmt>>(module.t));
  }

  // C1569
  void Post(const parser::SeparateModuleSubprogram &separateModuleSubprogram) {
    CheckOptionalName<parser::MpSubprogramStmt>("MODULE PROCEDURE",
        separateModuleSubprogram,
        std::get<parser::Statement<parser::EndMpSubprogramStmt>>(
            separateModuleSubprogram.t));
  }

  // C1401
  void Post(const parser::MainProgram &mainProgram) {
    if (const parser::CharBlock *
        endName{GetStmtName(std::get<parser::Statement<parser::EndProgramStmt>>(
            mainProgram.t))}) {
      if (const auto &program{
              std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(
                  mainProgram.t)}) {
        if (*endName != program->statement.v.source) {
          context_.Say(*endName, "END PROGRAM name mismatch"_err_en_US)
              .Attach(program->statement.v.source, "should be"_en_US);
        }
      } else {
        context_.Say(*endName,
            "END PROGRAM has name without PROGRAM statement"_err_en_US);
      }
    }
  }

  // C1413
  void Post(const parser::Submodule &submodule) {
    CheckOptionalName<parser::SubmoduleStmt>("SUBMODULE", submodule,
        std::get<parser::Statement<parser::EndSubmoduleStmt>>(submodule.t));
  }

  // C1567
  void Post(const parser::InterfaceBody::Subroutine &sub) {
    CheckOptionalName<parser::SubroutineStmt>("SUBROUTINE", sub,
        std::get<parser::Statement<parser::EndSubroutineStmt>>(sub.t));
  }

  // C1567
  void Post(const parser::SubroutineSubprogram &subroutineSubprogram) {
    CheckOptionalName<parser::SubroutineStmt>("SUBROUTINE",
        subroutineSubprogram,
        std::get<parser::Statement<parser::EndSubroutineStmt>>(
            subroutineSubprogram.t));
  }

  // C739
  bool Pre(const parser::DerivedTypeDef &) {
    PushDisposableMap();
    return true;
  }
  void Post(const parser::DerivedTypeDef &derivedTypeDef) {
    CheckOptionalName<parser::DerivedTypeStmt>("derived type definition",
        derivedTypeDef,
        std::get<parser::Statement<parser::EndTypeStmt>>(derivedTypeDef.t));
    PopDisposableMap();
  }

  void Post(const parser::LabelDoStmt &labelDoStmt) {
    AddLabelReferenceFromDoStmt(std::get<parser::Label>(labelDoStmt.t));
  }
  void Post(const parser::GotoStmt &gotoStmt) { AddLabelReference(gotoStmt.v); }
  void Post(const parser::ComputedGotoStmt &computedGotoStmt) {
    AddLabelReference(std::get<std::list<parser::Label>>(computedGotoStmt.t));
  }
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    AddLabelReference(std::get<1>(arithmeticIfStmt.t));
    AddLabelReference(std::get<2>(arithmeticIfStmt.t));
    AddLabelReference(std::get<3>(arithmeticIfStmt.t));
  }
  void Post(const parser::AssignStmt &assignStmt) {
    AddLabelReferenceFromAssignStmt(std::get<parser::Label>(assignStmt.t));
  }
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
    AddLabelReference(std::get<std::list<parser::Label>>(assignedGotoStmt.t));
  }
  void Post(const parser::AltReturnSpec &altReturnSpec) {
    AddLabelReference(altReturnSpec.v);
  }

  void Post(const parser::ErrLabel &errLabel) { AddLabelReference(errLabel.v); }
  void Post(const parser::EndLabel &endLabel) { AddLabelReference(endLabel.v); }
  void Post(const parser::EorLabel &eorLabel) { AddLabelReference(eorLabel.v); }
  void Post(const parser::Format &format) {
    if (const auto *labelPointer{std::get_if<parser::Label>(&format.u)}) {
      AddLabelReferenceToFormatStmt(*labelPointer);
    }
  }
  void Post(const parser::CycleStmt &cycleStmt) {
    if (cycleStmt.v) {
      CheckLabelContext("CYCLE", cycleStmt.v->source);
    }
  }
  void Post(const parser::ExitStmt &exitStmt) {
    if (exitStmt.v) {
      CheckLabelContext("EXIT", exitStmt.v->source);
    }
  }

  const std::vector<UnitAnalysis> &ProgramUnits() const {
    return programUnits_;
  }
  SemanticsContext &ErrorHandler() { return context_; }

private:
  ScopeInfo &PushScope(bool isExteriorGotoFatal) {
    auto &model{programUnits_.back().scopeModel};
    int newDepth{model.empty() ? 1 : model[currentScope_].depth + 1};
    ScopeInfo &result{model.emplace_back()};
    result.parent = currentScope_;
    result.depth = newDepth;
    result.isExteriorGotoFatal = isExteriorGotoFatal;
    currentScope_ = model.size() - 1;
    return result;
  }
  bool InitializeNewScopeContext() {
    programUnits_.emplace_back(UnitAnalysis{});
    currentScope_ = 0u;
    PushScope(false);
    return true;
  }
  ScopeInfo &PopScope() {
    ScopeInfo &result{programUnits_.back().scopeModel[currentScope_]};
    currentScope_ = result.parent;
    return result;
  }
  ProxyForScope ParentScope() {
    return programUnits_.back().scopeModel[currentScope_].parent;
  }
  bool SwitchToNewScope() {
    PushScope(PopScope().isExteriorGotoFatal);
    return true;
  }

  template <typename A> bool PushConstructName(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    // Gotos into this construct from outside it are diagnosed, and
    // are fatal unless the construct is a DO, IF, or SELECT CASE.
    PushScope(!(std::is_same_v<A, parser::DoConstruct> ||
        std::is_same_v<A, parser::IfConstruct> ||
        std::is_same_v<A, parser::CaseConstruct>));
    return true;
  }
  bool PushConstructName(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    PushScope(true);
    return true;
  }
  template <typename A> void PopConstructNameIfPresent(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName) {
      constructNames_.pop_back();
    }
  }
  void PopConstructNameIfPresent(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName) {
      constructNames_.pop_back();
    }
  }

  template <typename A> void PopConstructName(const A &a) {
    CheckName(a);
    PopScope();
    PopConstructNameIfPresent(a);
  }

  template <typename FIRST, typename CASEBLOCK, typename CASE,
      typename CONSTRUCT>
  void CheckSelectNames(const char *tag, const CONSTRUCT &construct) {
    CheckEndName<FIRST, parser::EndSelectStmt>(tag, construct);
    for (const auto &inner : std::get<std::list<CASEBLOCK>>(construct.t)) {
      CheckOptionalName<FIRST>(
          tag, construct, std::get<parser::Statement<CASE>>(inner.t));
    }
  }

  // C1144
  void PopConstructName(const parser::CaseConstruct &caseConstruct) {
    CheckSelectNames<parser::SelectCaseStmt, parser::CaseConstruct::Case,
        parser::CaseStmt>("SELECT CASE", caseConstruct);
    PopScope();
    PopConstructNameIfPresent(caseConstruct);
  }

  // C1154, C1156
  void PopConstructName(
      const parser::SelectRankConstruct &selectRankConstruct) {
    CheckSelectNames<parser::SelectRankStmt,
        parser::SelectRankConstruct::RankCase, parser::SelectRankCaseStmt>(
        "SELECT RANK", selectRankConstruct);
    PopScope();
    PopConstructNameIfPresent(selectRankConstruct);
  }

  // C1165
  void PopConstructName(
      const parser::SelectTypeConstruct &selectTypeConstruct) {
    CheckSelectNames<parser::SelectTypeStmt,
        parser::SelectTypeConstruct::TypeCase, parser::TypeGuardStmt>(
        "SELECT TYPE", selectTypeConstruct);
    PopScope();
    PopConstructNameIfPresent(selectTypeConstruct);
  }

  // Checks for missing or mismatching names on various constructs (e.g., BLOCK)
  // and their END statements.  Both names must be present if either one is.
  template <typename FIRST, typename END, typename CONSTRUCT>
  void CheckEndName(const char *constructTag, const CONSTRUCT &a) {
    const auto &constructStmt{std::get<parser::Statement<FIRST>>(a.t)};
    const auto &endStmt{std::get<parser::Statement<END>>(a.t)};
    const parser::CharBlock *endName{GetStmtName(endStmt)};
    if (const parser::CharBlock * constructName{GetStmtName(constructStmt)}) {
      if (endName) {
        if (*constructName != *endName) {
          context_
              .Say(*endName, "%s construct name mismatch"_err_en_US,
                  constructTag)
              .Attach(*constructName, "should be"_en_US);
        }
      } else {
        context_
            .Say(endStmt.source,
                "%s construct name required but missing"_err_en_US,
                constructTag)
            .Attach(*constructName, "should be"_en_US);
      }
    } else if (endName) {
      context_
          .Say(*endName, "%s construct name unexpected"_err_en_US, constructTag)
          .Attach(
              constructStmt.source, "unnamed %s statement"_en_US, constructTag);
    }
  }

  // C1106
  void CheckName(const parser::AssociateConstruct &associateConstruct) {
    CheckEndName<parser::AssociateStmt, parser::EndAssociateStmt>(
        "ASSOCIATE", associateConstruct);
  }
  // C1117
  void CheckName(const parser::CriticalConstruct &criticalConstruct) {
    CheckEndName<parser::CriticalStmt, parser::EndCriticalStmt>(
        "CRITICAL", criticalConstruct);
  }
  // C1131
  void CheckName(const parser::DoConstruct &doConstruct) {
    CheckEndName<parser::NonLabelDoStmt, parser::EndDoStmt>("DO", doConstruct);
    if (auto label{std::get<std::optional<parser::Label>>(
            std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)
                .statement.t)}) {
      const auto &endDoStmt{
          std::get<parser::Statement<parser::EndDoStmt>>(doConstruct.t)};
      if (!endDoStmt.label || *endDoStmt.label != *label) {
        context_
            .Say(endDoStmt.source,
                "END DO statement must have the label '%d' matching its DO statement"_err_en_US,
                *label)
            .Attach(std::get<parser::Statement<parser::NonLabelDoStmt>>(
                        doConstruct.t)
                        .source,
                "corresponding DO statement"_en_US);
      }
    }
  }
  // C1035
  void CheckName(const parser::ForallConstruct &forallConstruct) {
    CheckEndName<parser::ForallConstructStmt, parser::EndForallStmt>(
        "FORALL", forallConstruct);
  }

  // C1109
  void CheckName(const parser::BlockConstruct &blockConstruct) {
    CheckEndName<parser::BlockStmt, parser::EndBlockStmt>(
        "BLOCK", blockConstruct);
  }
  // C1112
  void CheckName(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    CheckEndName<parser::ChangeTeamStmt, parser::EndChangeTeamStmt>(
        "CHANGE TEAM", changeTeamConstruct);
  }

  // C1142
  void CheckName(const parser::IfConstruct &ifConstruct) {
    CheckEndName<parser::IfThenStmt, parser::EndIfStmt>("IF", ifConstruct);
    for (const auto &elseIfBlock :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(ifConstruct.t)) {
      CheckOptionalName<parser::IfThenStmt>("IF construct", ifConstruct,
          std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t));
    }
    if (const auto &elseBlock{
            std::get<std::optional<parser::IfConstruct::ElseBlock>>(
                ifConstruct.t)}) {
      CheckOptionalName<parser::IfThenStmt>("IF construct", ifConstruct,
          std::get<parser::Statement<parser::ElseStmt>>(elseBlock->t));
    }
  }

  // C1033
  void CheckName(const parser::WhereConstruct &whereConstruct) {
    CheckEndName<parser::WhereConstructStmt, parser::EndWhereStmt>(
        "WHERE", whereConstruct);
    for (const auto &maskedElsewhere :
        std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(
            whereConstruct.t)) {
      CheckOptionalName<parser::WhereConstructStmt>("WHERE construct",
          whereConstruct,
          std::get<parser::Statement<parser::MaskedElsewhereStmt>>(
              maskedElsewhere.t));
    }
    if (const auto &elsewhere{
            std::get<std::optional<parser::WhereConstruct::Elsewhere>>(
                whereConstruct.t)}) {
      CheckOptionalName<parser::WhereConstructStmt>("WHERE construct",
          whereConstruct,
          std::get<parser::Statement<parser::ElsewhereStmt>>(elsewhere->t));
    }
  }

  // C1134, C1166
  void CheckLabelContext(
      const char *const stmtString, const parser::CharBlock &constructName) {
    const auto iter{std::find(constructNames_.crbegin(),
        constructNames_.crend(), constructName.ToString())};
    if (iter == constructNames_.crend()) {
      context_.Say(constructName, "%s construct-name is not in scope"_err_en_US,
          stmtString);
    }
  }

  // 6.2.5, paragraph 2
  void CheckLabelInRange(parser::Label label) {
    if (label < 1 || label > 99999) {
      context_.Say(currentPosition_, "Label '%u' is out of range"_err_en_US,
          SayLabel(label));
    }
  }

  // 6.2.5., paragraph 2
  void AddTargetLabelDefinition(parser::Label label,
      LabeledStmtClassificationSet labeledStmtClassificationSet,
      ProxyForScope scope, bool isExecutableConstructEndStmt) {
    CheckLabelInRange(label);
    TargetStmtMap &targetStmtMap{disposableMaps_.empty()
            ? programUnits_.back().targetStmts
            : disposableMaps_.back()};
    const auto pair{targetStmtMap.emplace(label,
        LabeledStatementInfoTuplePOD{scope, currentPosition_,
            labeledStmtClassificationSet, isExecutableConstructEndStmt})};
    if (!pair.second) {
      context_.Say(currentPosition_, "Label '%u' is not distinct"_err_en_US,
          SayLabel(label));
    }
  }

  void AddLabelReferenceFromDoStmt(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().doStmtSources.emplace_back(
        label, currentScope_, currentPosition_);
  }

  void AddLabelReferenceToFormatStmt(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().formatStmtSources.emplace_back(
        label, currentScope_, currentPosition_);
  }

  void AddLabelReferenceFromAssignStmt(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().assignStmtSources.emplace_back(
        label, currentScope_, currentPosition_);
  }

  void AddLabelReference(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().otherStmtSources.emplace_back(
        label, currentScope_, currentPosition_);
  }

  void AddLabelReference(const std::list<parser::Label> &labels) {
    for (const parser::Label &label : labels) {
      AddLabelReference(label);
    }
  }

  void PushDisposableMap() { disposableMaps_.emplace_back(); }
  void PopDisposableMap() { disposableMaps_.pop_back(); }

  std::vector<UnitAnalysis> programUnits_;
  SemanticsContext &context_;
  parser::CharBlock currentPosition_;
  ProxyForScope currentScope_;
  std::vector<std::string> constructNames_;
  // For labels in derived type definitions and procedure
  // interfaces, which are their own inclusive scopes.  None
  // of these labels can be used as a branch target, but they
  // should be pairwise distinct.
  std::vector<TargetStmtMap> disposableMaps_;
};

bool InInclusiveScope(const std::vector<ScopeInfo> &scopes, ProxyForScope tail,
    ProxyForScope head) {
  for (; tail != head; tail = scopes[tail].parent) {
    if (!HasScope(tail)) {
      return false;
    }
  }
  return true;
}

ParseTreeAnalyzer LabelAnalysis(
    SemanticsContext &context, const parser::Program &program) {
  ParseTreeAnalyzer analysis{context};
  Walk(program, analysis);
  return analysis;
}

bool InBody(const parser::CharBlock &position,
    const std::pair<parser::CharBlock, parser::CharBlock> &pair) {
  if (position.begin() >= pair.first.begin()) {
    if (position.begin() < pair.second.end()) {
      return true;
    }
  }
  return false;
}

static LabeledStatementInfoTuplePOD GetLabel(
    const TargetStmtMap &labels, const parser::Label &label) {
  const auto iter{labels.find(label)};
  if (iter == labels.cend()) {
    return {0u, nullptr, LabeledStmtClassificationSet{}, false};
  } else {
    return iter->second;
  }
}

// 11.1.7.3
void CheckBranchesIntoDoBody(const SourceStmtList &branches,
    const TargetStmtMap &labels, const IndexList &loopBodies,
    SemanticsContext &context) {
  for (const auto &branch : branches) {
    const auto &label{branch.parserLabel};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(branchTarget.proxyForScope)) {
      const auto &fromPosition{branch.parserCharBlock};
      const auto &toPosition{branchTarget.parserCharBlock};
      for (const auto &body : loopBodies) {
        if (!InBody(fromPosition, body) && InBody(toPosition, body) &&
            context.ShouldWarn(common::LanguageFeature::BranchIntoConstruct)) {
          context
              .Say(
                  fromPosition, "branch into loop body from outside"_warn_en_US)
              .Attach(body.first, "the loop branched into"_en_US)
              .set_languageFeature(
                  common::LanguageFeature::BranchIntoConstruct);
        }
      }
    }
  }
}

void CheckDoNesting(const IndexList &loopBodies, SemanticsContext &context) {
  for (auto i1{loopBodies.cbegin()}; i1 != loopBodies.cend(); ++i1) {
    const auto &v1{*i1};
    for (auto i2{i1 + 1}; i2 != loopBodies.cend(); ++i2) {
      const auto &v2{*i2};
      if (v2.first.begin() < v1.second.end() &&
          v1.second.begin() < v2.second.begin()) {
        context.Say(v1.first, "DO loop doesn't properly nest"_err_en_US)
            .Attach(v2.first, "DO loop conflicts"_en_US);
      }
    }
  }
}

parser::CharBlock SkipLabel(const parser::CharBlock &position) {
  const std::size_t maxPosition{position.size()};
  if (maxPosition && parser::IsDecimalDigit(position[0])) {
    std::size_t i{1l};
    for (; (i < maxPosition) && parser::IsDecimalDigit(position[i]); ++i) {
    }
    for (; (i < maxPosition) && parser::IsWhiteSpace(position[i]); ++i) {
    }
    return parser::CharBlock{position.begin() + i, position.end()};
  }
  return position;
}

ProxyForScope ParentScope(
    const std::vector<ScopeInfo> &scopes, ProxyForScope scope) {
  return scopes[scope].parent;
}

void CheckLabelDoConstraints(const SourceStmtList &dos,
    const SourceStmtList &branches, const TargetStmtMap &labels,
    const std::vector<ScopeInfo> &scopes, SemanticsContext &context) {
  IndexList loopBodies;
  for (const auto &stmt : dos) {
    const auto &label{stmt.parserLabel};
    const auto &scope{stmt.proxyForScope};
    const auto &position{stmt.parserCharBlock};
    auto doTarget{GetLabel(labels, label)};
    if (!HasScope(doTarget.proxyForScope)) {
      // C1133
      context.Say(
          position, "Label '%u' cannot be found"_err_en_US, SayLabel(label));
    } else if (doTarget.parserCharBlock.begin() < position.begin()) {
      // R1119
      context.Say(position,
          "Label '%u' doesn't lexically follow DO stmt"_err_en_US,
          SayLabel(label));

    } else if ((InInclusiveScope(scopes, scope, doTarget.proxyForScope) &&
                   doTarget.labeledStmtClassificationSet.test(
                       TargetStatementEnum::CompatibleDo)) ||
        (doTarget.isExecutableConstructEndStmt &&
            ParentScope(scopes, doTarget.proxyForScope) == scope)) {
      if (context.ShouldWarn(
              common::LanguageFeature::OldLabelDoEndStatements)) {
        context
            .Say(position,
                "A DO loop should terminate with an END DO or CONTINUE"_port_en_US)
            .Attach(doTarget.parserCharBlock,
                "DO loop currently ends at statement:"_en_US)
            .set_languageFeature(
                common::LanguageFeature::OldLabelDoEndStatements);
      }
    } else if (!InInclusiveScope(scopes, scope, doTarget.proxyForScope)) {
      context.Say(position, "Label '%u' is not in DO loop scope"_err_en_US,
          SayLabel(label));
    } else if (!doTarget.labeledStmtClassificationSet.test(
                   TargetStatementEnum::Do)) {
      context.Say(doTarget.parserCharBlock,
          "A DO loop should terminate with an END DO or CONTINUE"_err_en_US);
    } else {
      loopBodies.emplace_back(SkipLabel(position), doTarget.parserCharBlock);
    }
  }

  CheckBranchesIntoDoBody(branches, labels, loopBodies, context);
  CheckDoNesting(loopBodies, context);
}

// 6.2.5
void CheckScopeConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, const std::vector<ScopeInfo> &scopes,
    SemanticsContext &context) {
  for (const auto &stmt : stmts) {
    const auto &label{stmt.parserLabel};
    const auto &scope{stmt.proxyForScope};
    const auto &position{stmt.parserCharBlock};
    auto target{GetLabel(labels, label)};
    if (!HasScope(target.proxyForScope)) {
      context.Say(
          position, "Label '%u' was not found"_err_en_US, SayLabel(label));
    } else if (!InInclusiveScope(scopes, scope, target.proxyForScope)) {
      // Clause 11.1.2.1 prohibits transfer of control to the interior of a
      // block from outside the block, but this does not apply to formats.
      // C1038 and C1034 forbid statements in FORALL and WHERE constructs
      // (resp.) from being branch targets.
      if (target.labeledStmtClassificationSet.test(
              TargetStatementEnum::Format)) {
        continue;
      }
      bool isFatal{false};
      ProxyForScope fromScope{scope};
      for (ProxyForScope toScope{target.proxyForScope}; HasScope(toScope);
           toScope = scopes[toScope].parent) {
        while (scopes[fromScope].depth > scopes[toScope].depth) {
          fromScope = scopes[fromScope].parent;
        }
        if (toScope == fromScope) {
          break;
        }
        if (scopes[toScope].isExteriorGotoFatal) {
          isFatal = true;
          break;
        }
      }
      if (isFatal) {
        context.Say(position,
            "Label '%u' is in a construct that prevents its use as a branch target here"_err_en_US,
            SayLabel(label));
      } else if (context.ShouldWarn(
                     common::LanguageFeature::BranchIntoConstruct)) {
        context
            .Say(position,
                "Label '%u' is in a construct that should not be used as a branch target here"_warn_en_US,
                SayLabel(label))
            .set_languageFeature(common::LanguageFeature::BranchIntoConstruct);
      }
    }
  }
}

void CheckBranchTargetConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, SemanticsContext &context) {
  for (const auto &stmt : stmts) {
    const auto &label{stmt.parserLabel};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(branchTarget.proxyForScope)) {
      if (!branchTarget.labeledStmtClassificationSet.test(
              TargetStatementEnum::Branch) &&
          !branchTarget.labeledStmtClassificationSet.test(
              TargetStatementEnum::CompatibleBranch)) { // error
        context
            .Say(branchTarget.parserCharBlock,
                "Label '%u' is not a branch target"_err_en_US, SayLabel(label))
            .Attach(stmt.parserCharBlock, "Control flow use of '%u'"_en_US,
                SayLabel(label));
      } else if (!branchTarget.labeledStmtClassificationSet.test(
                     TargetStatementEnum::Branch) &&
          context.ShouldWarn(common::LanguageFeature::BadBranchTarget)) {
        context
            .Say(branchTarget.parserCharBlock,
                "Label '%u' is not a branch target"_warn_en_US, SayLabel(label))
            .Attach(stmt.parserCharBlock, "Control flow use of '%u'"_en_US,
                SayLabel(label))
            .set_languageFeature(common::LanguageFeature::BadBranchTarget);
      }
    }
  }
}

void CheckBranchConstraints(const SourceStmtList &branches,
    const TargetStmtMap &labels, const std::vector<ScopeInfo> &scopes,
    SemanticsContext &context) {
  CheckScopeConstraints(branches, labels, scopes, context);
  CheckBranchTargetConstraints(branches, labels, context);
}

void CheckDataXferTargetConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, SemanticsContext &context) {
  for (const auto &stmt : stmts) {
    const auto &label{stmt.parserLabel};
    auto ioTarget{GetLabel(labels, label)};
    if (HasScope(ioTarget.proxyForScope)) {
      if (!ioTarget.labeledStmtClassificationSet.test(
              TargetStatementEnum::Format)) {
        context
            .Say(ioTarget.parserCharBlock, "'%u' not a FORMAT"_err_en_US,
                SayLabel(label))
            .Attach(stmt.parserCharBlock, "data transfer use of '%u'"_en_US,
                SayLabel(label));
      }
    }
  }
}

void CheckDataTransferConstraints(const SourceStmtList &dataTransfers,
    const TargetStmtMap &labels, const std::vector<ScopeInfo> &scopes,
    SemanticsContext &context) {
  CheckScopeConstraints(dataTransfers, labels, scopes, context);
  CheckDataXferTargetConstraints(dataTransfers, labels, context);
}

void CheckAssignTargetConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, SemanticsContext &context) {
  for (const auto &stmt : stmts) {
    const auto &label{stmt.parserLabel};
    auto target{GetLabel(labels, label)};
    if (HasScope(target.proxyForScope) &&
        !target.labeledStmtClassificationSet.test(
            TargetStatementEnum::Branch) &&
        !target.labeledStmtClassificationSet.test(
            TargetStatementEnum::Format)) {
      parser::Message *msg{nullptr};
      if (!target.labeledStmtClassificationSet.test(
              TargetStatementEnum::CompatibleBranch)) {
        msg = &context.Say(target.parserCharBlock,
            "Label '%u' is not a branch target or FORMAT"_err_en_US,
            SayLabel(label));
      } else if (context.ShouldWarn(common::LanguageFeature::BadBranchTarget)) {
        msg =
            &context
                 .Say(target.parserCharBlock,
                     "Label '%u' is not a branch target or FORMAT"_warn_en_US,
                     SayLabel(label))
                 .set_languageFeature(common::LanguageFeature::BadBranchTarget);
      }
      if (msg) {
        msg->Attach(stmt.parserCharBlock, "ASSIGN statement use of '%u'"_en_US,
            SayLabel(label));
      }
    }
  }
}

void CheckAssignConstraints(const SourceStmtList &assigns,
    const TargetStmtMap &labels, const std::vector<ScopeInfo> &scopes,
    SemanticsContext &context) {
  CheckScopeConstraints(assigns, labels, scopes, context);
  CheckAssignTargetConstraints(assigns, labels, context);
}

bool CheckConstraints(ParseTreeAnalyzer &&parseTreeAnalysis) {
  auto &context{parseTreeAnalysis.ErrorHandler()};
  for (const auto &programUnit : parseTreeAnalysis.ProgramUnits()) {
    const auto &dos{programUnit.doStmtSources};
    const auto &branches{programUnit.otherStmtSources};
    const auto &labels{programUnit.targetStmts};
    const auto &scopes{programUnit.scopeModel};
    CheckLabelDoConstraints(dos, branches, labels, scopes, context);
    CheckBranchConstraints(branches, labels, scopes, context);
    const auto &dataTransfers{programUnit.formatStmtSources};
    CheckDataTransferConstraints(dataTransfers, labels, scopes, context);
    const auto &assigns{programUnit.assignStmtSources};
    CheckAssignConstraints(assigns, labels, scopes, context);
  }
  return !context.AnyFatalError();
}

bool ValidateLabels(SemanticsContext &context, const parser::Program &program) {
  return CheckConstraints(LabelAnalysis(context, program));
}
} // namespace Fortran::semantics
