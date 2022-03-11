//===-- lib/Semantics/check-do-forall.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-do-forall.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/call.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

namespace Fortran::evaluate {
using ActualArgumentRef = common::Reference<const ActualArgument>;

inline bool operator<(ActualArgumentRef x, ActualArgumentRef y) {
  return &*x < &*y;
}
} // namespace Fortran::evaluate

namespace Fortran::semantics {

using namespace parser::literals;

using Bounds = parser::LoopControl::Bounds;
using IndexVarKind = SemanticsContext::IndexVarKind;

static const parser::ConcurrentHeader &GetConcurrentHeader(
    const parser::LoopControl &loopControl) {
  const auto &concurrent{
      std::get<parser::LoopControl::Concurrent>(loopControl.u)};
  return std::get<parser::ConcurrentHeader>(concurrent.t);
}
static const parser::ConcurrentHeader &GetConcurrentHeader(
    const parser::ForallConstruct &construct) {
  const auto &stmt{
      std::get<parser::Statement<parser::ForallConstructStmt>>(construct.t)};
  return std::get<common::Indirection<parser::ConcurrentHeader>>(
      stmt.statement.t)
      .value();
}
static const parser::ConcurrentHeader &GetConcurrentHeader(
    const parser::ForallStmt &stmt) {
  return std::get<common::Indirection<parser::ConcurrentHeader>>(stmt.t)
      .value();
}
template <typename T>
static const std::list<parser::ConcurrentControl> &GetControls(const T &x) {
  return std::get<std::list<parser::ConcurrentControl>>(
      GetConcurrentHeader(x).t);
}

static const Bounds &GetBounds(const parser::DoConstruct &doConstruct) {
  auto &loopControl{doConstruct.GetLoopControl().value()};
  return std::get<Bounds>(loopControl.u);
}

static const parser::Name &GetDoVariable(
    const parser::DoConstruct &doConstruct) {
  const Bounds &bounds{GetBounds(doConstruct)};
  return bounds.name.thing;
}

static parser::MessageFixedText GetEnclosingDoMsg() {
  return "Enclosing DO CONCURRENT statement"_en_US;
}

static void SayWithDo(SemanticsContext &context, parser::CharBlock stmtLocation,
    parser::MessageFixedText &&message, parser::CharBlock doLocation) {
  context.Say(stmtLocation, message).Attach(doLocation, GetEnclosingDoMsg());
}

// 11.1.7.5 - enforce semantics constraints on a DO CONCURRENT loop body
class DoConcurrentBodyEnforce {
public:
  DoConcurrentBodyEnforce(
      SemanticsContext &context, parser::CharBlock doConcurrentSourcePosition)
      : context_{context}, doConcurrentSourcePosition_{
                               doConcurrentSourcePosition} {}
  std::set<parser::Label> labels() { return labels_; }
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  template <typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    if (statement.label.has_value()) {
      labels_.insert(*statement.label);
    }
    return true;
  }

  template <typename T> bool Pre(const parser::UnlabeledStatement<T> &stmt) {
    currentStatementSourcePosition_ = stmt.source;
    return true;
  }

  // C1140 -- Can't deallocate a polymorphic entity in a DO CONCURRENT.
  // Deallocation can be caused by exiting a block that declares an allocatable
  // entity, assignment to an allocatable variable, or an actual DEALLOCATE
  // statement
  //
  // Note also that the deallocation of a derived type entity might cause the
  // invocation of an IMPURE final subroutine. (C1139)
  //

  // Only to be called for symbols with ObjectEntityDetails
  static bool HasImpureFinal(const Symbol &original) {
    const Symbol &symbol{ResolveAssociations(original)};
    if (symbol.has<ObjectEntityDetails>()) {
      if (const DeclTypeSpec * symType{symbol.GetType()}) {
        if (const DerivedTypeSpec * derived{symType->AsDerived()}) {
          return semantics::HasImpureFinal(*derived);
        }
      }
    }
    return false;
  }

  // Predicate for deallocations caused by block exit and direct deallocation
  static bool DeallocateAll(const Symbol &) { return true; }

  // Predicate for deallocations caused by intrinsic assignment
  static bool DeallocateNonCoarray(const Symbol &component) {
    return !evaluate::IsCoarray(component);
  }

  static bool WillDeallocatePolymorphic(const Symbol &entity,
      const std::function<bool(const Symbol &)> &WillDeallocate) {
    return WillDeallocate(entity) && IsPolymorphicAllocatable(entity);
  }

  // Is it possible that we will we deallocate a polymorphic entity or one
  // of its components?
  static bool MightDeallocatePolymorphic(const Symbol &original,
      const std::function<bool(const Symbol &)> &WillDeallocate) {
    const Symbol &symbol{ResolveAssociations(original)};
    // Check the entity itself, no coarray exception here
    if (IsPolymorphicAllocatable(symbol)) {
      return true;
    }
    // Check the components
    if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * entityType{details->type()}) {
        if (const DerivedTypeSpec * derivedType{entityType->AsDerived()}) {
          UltimateComponentIterator ultimates{*derivedType};
          for (const auto &ultimate : ultimates) {
            if (WillDeallocatePolymorphic(ultimate, WillDeallocate)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  void SayDeallocateWithImpureFinal(const Symbol &entity, const char *reason) {
    context_.SayWithDecl(entity, currentStatementSourcePosition_,
        "Deallocation of an entity with an IMPURE FINAL procedure"
        " caused by %s not allowed in DO CONCURRENT"_err_en_US,
        reason);
  }

  void SayDeallocateOfPolymorph(
      parser::CharBlock location, const Symbol &entity, const char *reason) {
    context_.SayWithDecl(entity, location,
        "Deallocation of a polymorphic entity caused by %s"
        " not allowed in DO CONCURRENT"_err_en_US,
        reason);
  }

  // Deallocation caused by block exit
  // Allocatable entities and all of their allocatable subcomponents will be
  // deallocated.  This test is different from the other two because it does
  // not deallocate in cases where the entity itself is not allocatable but
  // has allocatable polymorphic components
  void Post(const parser::BlockConstruct &blockConstruct) {
    const auto &endBlockStmt{
        std::get<parser::Statement<parser::EndBlockStmt>>(blockConstruct.t)};
    const Scope &blockScope{context_.FindScope(endBlockStmt.source)};
    const Scope &doScope{context_.FindScope(doConcurrentSourcePosition_)};
    if (DoesScopeContain(&doScope, blockScope)) {
      const char *reason{"block exit"};
      for (auto &pair : blockScope) {
        const Symbol &entity{*pair.second};
        if (IsAllocatable(entity) && !IsSaved(entity) &&
            MightDeallocatePolymorphic(entity, DeallocateAll)) {
          SayDeallocateOfPolymorph(endBlockStmt.source, entity, reason);
        }
        if (HasImpureFinal(entity)) {
          SayDeallocateWithImpureFinal(entity, reason);
        }
      }
    }
  }

  // Deallocation caused by assignment
  // Note that this case does not cause deallocation of coarray components
  void Post(const parser::AssignmentStmt &stmt) {
    const auto &variable{std::get<parser::Variable>(stmt.t)};
    if (const Symbol * entity{GetLastName(variable).symbol}) {
      const char *reason{"assignment"};
      if (MightDeallocatePolymorphic(*entity, DeallocateNonCoarray)) {
        SayDeallocateOfPolymorph(variable.GetSource(), *entity, reason);
      }
      if (HasImpureFinal(*entity)) {
        SayDeallocateWithImpureFinal(*entity, reason);
      }
    }
  }

  // Deallocation from a DEALLOCATE statement
  // This case is different because DEALLOCATE statements deallocate both
  // ALLOCATABLE and POINTER entities
  void Post(const parser::DeallocateStmt &stmt) {
    const auto &allocateObjectList{
        std::get<std::list<parser::AllocateObject>>(stmt.t)};
    for (const auto &allocateObject : allocateObjectList) {
      const parser::Name &name{GetLastName(allocateObject)};
      const char *reason{"a DEALLOCATE statement"};
      if (name.symbol) {
        const Symbol &entity{*name.symbol};
        const DeclTypeSpec *entityType{entity.GetType()};
        if ((entityType && entityType->IsPolymorphic()) || // POINTER case
            MightDeallocatePolymorphic(entity, DeallocateAll)) {
          SayDeallocateOfPolymorph(
              currentStatementSourcePosition_, entity, reason);
        }
        if (HasImpureFinal(entity)) {
          SayDeallocateWithImpureFinal(entity, reason);
        }
      }
    }
  }

  // C1137 -- No image control statements in a DO CONCURRENT
  void Post(const parser::ExecutableConstruct &construct) {
    if (IsImageControlStmt(construct)) {
      const parser::CharBlock statementLocation{
          GetImageControlStmtLocation(construct)};
      auto &msg{context_.Say(statementLocation,
          "An image control statement is not allowed in DO"
          " CONCURRENT"_err_en_US)};
      if (auto coarrayMsg{GetImageControlStmtCoarrayMsg(construct)}) {
        msg.Attach(statementLocation, *coarrayMsg);
      }
      msg.Attach(doConcurrentSourcePosition_, GetEnclosingDoMsg());
    }
  }

  // C1136 -- No RETURN statements in a DO CONCURRENT
  void Post(const parser::ReturnStmt &) {
    context_
        .Say(currentStatementSourcePosition_,
            "RETURN is not allowed in DO CONCURRENT"_err_en_US)
        .Attach(doConcurrentSourcePosition_, GetEnclosingDoMsg());
  }

  // C1139: call to impure procedure and ...
  // C1141: cannot call ieee_get_flag, ieee_[gs]et_halting_mode
  // It's not necessary to check the ieee_get* procedures because they're
  // not pure, and impure procedures are caught by checks for constraint C1139
  void Post(const parser::ProcedureDesignator &procedureDesignator) {
    if (auto *name{std::get_if<parser::Name>(&procedureDesignator.u)}) {
      if (name->symbol && !IsPureProcedure(*name->symbol)) {
        SayWithDo(context_, currentStatementSourcePosition_,
            "Call to an impure procedure is not allowed in DO"
            " CONCURRENT"_err_en_US,
            doConcurrentSourcePosition_);
      }
      if (name->symbol &&
          fromScope(*name->symbol, "__fortran_ieee_exceptions"s)) {
        if (name->source == "ieee_set_halting_mode") {
          SayWithDo(context_, currentStatementSourcePosition_,
              "IEEE_SET_HALTING_MODE is not allowed in DO "
              "CONCURRENT"_err_en_US,
              doConcurrentSourcePosition_);
        }
      }
    } else {
      // C1139: this a procedure component
      auto &component{std::get<parser::ProcComponentRef>(procedureDesignator.u)
                          .v.thing.component};
      if (component.symbol && !IsPureProcedure(*component.symbol)) {
        SayWithDo(context_, currentStatementSourcePosition_,
            "Call to an impure procedure component is not allowed"
            " in DO CONCURRENT"_err_en_US,
            doConcurrentSourcePosition_);
      }
    }
  }

  // 11.1.7.5, paragraph 5, no ADVANCE specifier in a DO CONCURRENT
  void Post(const parser::IoControlSpec &ioControlSpec) {
    if (auto *charExpr{
            std::get_if<parser::IoControlSpec::CharExpr>(&ioControlSpec.u)}) {
      if (std::get<parser::IoControlSpec::CharExpr::Kind>(charExpr->t) ==
          parser::IoControlSpec::CharExpr::Kind::Advance) {
        SayWithDo(context_, currentStatementSourcePosition_,
            "ADVANCE specifier is not allowed in DO"
            " CONCURRENT"_err_en_US,
            doConcurrentSourcePosition_);
      }
    }
  }

private:
  bool fromScope(const Symbol &symbol, const std::string &moduleName) {
    if (symbol.GetUltimate().owner().IsModule() &&
        symbol.GetUltimate().owner().GetName().value().ToString() ==
            moduleName) {
      return true;
    }
    return false;
  }

  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_;
  SemanticsContext &context_;
  parser::CharBlock doConcurrentSourcePosition_;
}; // class DoConcurrentBodyEnforce

// Class for enforcing C1130 -- in a DO CONCURRENT with DEFAULT(NONE),
// variables from enclosing scopes must have their locality specified
class DoConcurrentVariableEnforce {
public:
  DoConcurrentVariableEnforce(
      SemanticsContext &context, parser::CharBlock doConcurrentSourcePosition)
      : context_{context},
        doConcurrentSourcePosition_{doConcurrentSourcePosition},
        blockScope_{context.FindScope(doConcurrentSourcePosition_)} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  // Check to see if the name is a variable from an enclosing scope
  void Post(const parser::Name &name) {
    if (const Symbol * symbol{name.symbol}) {
      if (IsVariableName(*symbol)) {
        const Scope &variableScope{symbol->owner()};
        if (DoesScopeContain(&variableScope, blockScope_)) {
          context_.SayWithDecl(*symbol, name.source,
              "Variable '%s' from an enclosing scope referenced in DO "
              "CONCURRENT with DEFAULT(NONE) must appear in a "
              "locality-spec"_err_en_US,
              symbol->name());
        }
      }
    }
  }

private:
  SemanticsContext &context_;
  parser::CharBlock doConcurrentSourcePosition_;
  const Scope &blockScope_;
}; // class DoConcurrentVariableEnforce

// Find a DO or FORALL and enforce semantics checks on its body
class DoContext {
public:
  DoContext(SemanticsContext &context, IndexVarKind kind)
      : context_{context}, kind_{kind} {}

  // Mark this DO construct as a point of definition for the DO variables
  // or index-names it contains.  If they're already defined, emit an error
  // message.  We need to remember both the variable and the source location of
  // the variable in the DO construct so that we can remove it when we leave
  // the DO construct and use its location in error messages.
  void DefineDoVariables(const parser::DoConstruct &doConstruct) {
    if (doConstruct.IsDoNormal()) {
      context_.ActivateIndexVar(GetDoVariable(doConstruct), IndexVarKind::DO);
    } else if (doConstruct.IsDoConcurrent()) {
      if (const auto &loopControl{doConstruct.GetLoopControl()}) {
        ActivateIndexVars(GetControls(*loopControl));
      }
    }
  }

  // Called at the end of a DO construct to deactivate the DO construct
  void ResetDoVariables(const parser::DoConstruct &doConstruct) {
    if (doConstruct.IsDoNormal()) {
      context_.DeactivateIndexVar(GetDoVariable(doConstruct));
    } else if (doConstruct.IsDoConcurrent()) {
      if (const auto &loopControl{doConstruct.GetLoopControl()}) {
        DeactivateIndexVars(GetControls(*loopControl));
      }
    }
  }

  void ActivateIndexVars(const std::list<parser::ConcurrentControl> &controls) {
    for (const auto &control : controls) {
      context_.ActivateIndexVar(std::get<parser::Name>(control.t), kind_);
    }
  }
  void DeactivateIndexVars(
      const std::list<parser::ConcurrentControl> &controls) {
    for (const auto &control : controls) {
      context_.DeactivateIndexVar(std::get<parser::Name>(control.t));
    }
  }

  void Check(const parser::DoConstruct &doConstruct) {
    if (doConstruct.IsDoConcurrent()) {
      CheckDoConcurrent(doConstruct);
      return;
    }
    if (doConstruct.IsDoNormal()) {
      CheckDoNormal(doConstruct);
      return;
    }
    // TODO: handle the other cases
  }

  void Check(const parser::ForallStmt &stmt) {
    CheckConcurrentHeader(GetConcurrentHeader(stmt));
  }
  void Check(const parser::ForallConstruct &construct) {
    CheckConcurrentHeader(GetConcurrentHeader(construct));
  }

  void Check(const parser::ForallAssignmentStmt &stmt) {
    const evaluate::Assignment *assignment{std::visit(
        common::visitors{[&](const auto &x) { return GetAssignment(x); }},
        stmt.u)};
    if (assignment) {
      CheckForallIndexesUsed(*assignment);
      CheckForImpureCall(assignment->lhs);
      CheckForImpureCall(assignment->rhs);
      if (const auto *proc{
              std::get_if<evaluate::ProcedureRef>(&assignment->u)}) {
        CheckForImpureCall(*proc);
      }
      std::visit(common::visitors{
                     [](const evaluate::Assignment::Intrinsic &) {},
                     [&](const evaluate::ProcedureRef &proc) {
                       CheckForImpureCall(proc);
                     },
                     [&](const evaluate::Assignment::BoundsSpec &bounds) {
                       for (const auto &bound : bounds) {
                         CheckForImpureCall(SomeExpr{bound});
                       }
                     },
                     [&](const evaluate::Assignment::BoundsRemapping &bounds) {
                       for (const auto &bound : bounds) {
                         CheckForImpureCall(SomeExpr{bound.first});
                         CheckForImpureCall(SomeExpr{bound.second});
                       }
                     },
                 },
          assignment->u);
    }
  }

private:
  void SayBadDoControl(parser::CharBlock sourceLocation) {
    context_.Say(sourceLocation, "DO controls should be INTEGER"_err_en_US);
  }

  void CheckDoControl(const parser::CharBlock &sourceLocation, bool isReal) {
    const bool warn{context_.warnOnNonstandardUsage() ||
        context_.ShouldWarn(common::LanguageFeature::RealDoControls)};
    if (isReal && !warn) {
      // No messages for the default case
    } else if (isReal && warn) {
      context_.Say(sourceLocation, "DO controls should be INTEGER"_port_en_US);
    } else {
      SayBadDoControl(sourceLocation);
    }
  }

  void CheckDoVariable(const parser::ScalarName &scalarName) {
    const parser::CharBlock &sourceLocation{scalarName.thing.source};
    if (const Symbol * symbol{scalarName.thing.symbol}) {
      if (!IsVariableName(*symbol)) {
        context_.Say(
            sourceLocation, "DO control must be an INTEGER variable"_err_en_US);
      } else {
        const DeclTypeSpec *symType{symbol->GetType()};
        if (!symType) {
          SayBadDoControl(sourceLocation);
        } else {
          if (!symType->IsNumeric(TypeCategory::Integer)) {
            CheckDoControl(
                sourceLocation, symType->IsNumeric(TypeCategory::Real));
          }
        }
      } // No messages for INTEGER
    }
  }

  // Semantic checks for the limit and step expressions
  void CheckDoExpression(const parser::ScalarExpr &scalarExpression) {
    if (const SomeExpr * expr{GetExpr(scalarExpression)}) {
      if (!ExprHasTypeCategory(*expr, TypeCategory::Integer)) {
        // No warnings or errors for type INTEGER
        const parser::CharBlock &loc{scalarExpression.thing.value().source};
        CheckDoControl(loc, ExprHasTypeCategory(*expr, TypeCategory::Real));
      }
    }
  }

  void CheckDoNormal(const parser::DoConstruct &doConstruct) {
    // C1120 -- types of DO variables must be INTEGER, extended by allowing
    // REAL and DOUBLE PRECISION
    const Bounds &bounds{GetBounds(doConstruct)};
    CheckDoVariable(bounds.name);
    CheckDoExpression(bounds.lower);
    CheckDoExpression(bounds.upper);
    if (bounds.step) {
      CheckDoExpression(*bounds.step);
      if (IsZero(*bounds.step)) {
        context_.Say(bounds.step->thing.value().source,
            "DO step expression should not be zero"_warn_en_US);
      }
    }
  }

  void CheckDoConcurrent(const parser::DoConstruct &doConstruct) {
    auto &doStmt{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)};
    currentStatementSourcePosition_ = doStmt.source;

    const parser::Block &block{std::get<parser::Block>(doConstruct.t)};
    DoConcurrentBodyEnforce doConcurrentBodyEnforce{context_, doStmt.source};
    parser::Walk(block, doConcurrentBodyEnforce);

    LabelEnforce doConcurrentLabelEnforce{context_,
        doConcurrentBodyEnforce.labels(), currentStatementSourcePosition_,
        "DO CONCURRENT"};
    parser::Walk(block, doConcurrentLabelEnforce);

    const auto &loopControl{doConstruct.GetLoopControl()};
    CheckConcurrentLoopControl(*loopControl);
    CheckLocalitySpecs(*loopControl, block);
  }

  // Return a set of symbols whose names are in a Local locality-spec.  Look
  // the names up in the scope that encloses the DO construct to avoid getting
  // the local versions of them.  Then follow the host-, use-, and
  // construct-associations to get the root symbols
  UnorderedSymbolSet GatherLocals(
      const std::list<parser::LocalitySpec> &localitySpecs) const {
    UnorderedSymbolSet symbols;
    const Scope &parentScope{
        context_.FindScope(currentStatementSourcePosition_).parent()};
    // Loop through the LocalitySpec::Local locality-specs
    for (const auto &ls : localitySpecs) {
      if (const auto *names{std::get_if<parser::LocalitySpec::Local>(&ls.u)}) {
        // Loop through the names in the Local locality-spec getting their
        // symbols
        for (const parser::Name &name : names->v) {
          if (const Symbol * symbol{parentScope.FindSymbol(name.source)}) {
            symbols.insert(ResolveAssociations(*symbol));
          }
        }
      }
    }
    return symbols;
  }

  static UnorderedSymbolSet GatherSymbolsFromExpression(
      const parser::Expr &expression) {
    UnorderedSymbolSet result;
    if (const auto *expr{GetExpr(expression)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
        result.insert(ResolveAssociations(symbol));
      }
    }
    return result;
  }

  // C1121 - procedures in mask must be pure
  void CheckMaskIsPure(const parser::ScalarLogicalExpr &mask) const {
    UnorderedSymbolSet references{
        GatherSymbolsFromExpression(mask.thing.thing.value())};
    for (const Symbol &ref : OrderBySourcePosition(references)) {
      if (IsProcedure(ref) && !IsPureProcedure(ref)) {
        context_.SayWithDecl(ref, parser::Unwrap<parser::Expr>(mask)->source,
            "%s mask expression may not reference impure procedure '%s'"_err_en_US,
            LoopKindName(), ref.name());
        return;
      }
    }
  }

  void CheckNoCollisions(const UnorderedSymbolSet &refs,
      const UnorderedSymbolSet &uses, parser::MessageFixedText &&errorMessage,
      const parser::CharBlock &refPosition) const {
    for (const Symbol &ref : OrderBySourcePosition(refs)) {
      if (uses.find(ref) != uses.end()) {
        context_.SayWithDecl(ref, refPosition, std::move(errorMessage),
            LoopKindName(), ref.name());
        return;
      }
    }
  }

  void HasNoReferences(const UnorderedSymbolSet &indexNames,
      const parser::ScalarIntExpr &expr) const {
    CheckNoCollisions(GatherSymbolsFromExpression(expr.thing.thing.value()),
        indexNames,
        "%s limit expression may not reference index variable '%s'"_err_en_US,
        expr.thing.thing.value().source);
  }

  // C1129, names in local locality-specs can't be in mask expressions
  void CheckMaskDoesNotReferenceLocal(const parser::ScalarLogicalExpr &mask,
      const UnorderedSymbolSet &localVars) const {
    CheckNoCollisions(GatherSymbolsFromExpression(mask.thing.thing.value()),
        localVars,
        "%s mask expression references variable '%s'"
        " in LOCAL locality-spec"_err_en_US,
        mask.thing.thing.value().source);
  }

  // C1129, names in local locality-specs can't be in limit or step
  // expressions
  void CheckExprDoesNotReferenceLocal(const parser::ScalarIntExpr &expr,
      const UnorderedSymbolSet &localVars) const {
    CheckNoCollisions(GatherSymbolsFromExpression(expr.thing.thing.value()),
        localVars,
        "%s expression references variable '%s'"
        " in LOCAL locality-spec"_err_en_US,
        expr.thing.thing.value().source);
  }

  // C1130, DEFAULT(NONE) locality requires names to be in locality-specs to
  // be used in the body of the DO loop
  void CheckDefaultNoneImpliesExplicitLocality(
      const std::list<parser::LocalitySpec> &localitySpecs,
      const parser::Block &block) const {
    bool hasDefaultNone{false};
    for (auto &ls : localitySpecs) {
      if (std::holds_alternative<parser::LocalitySpec::DefaultNone>(ls.u)) {
        if (hasDefaultNone) {
          // C1127, you can only have one DEFAULT(NONE)
          context_.Say(currentStatementSourcePosition_,
              "Only one DEFAULT(NONE) may appear"_port_en_US);
          break;
        }
        hasDefaultNone = true;
      }
    }
    if (hasDefaultNone) {
      DoConcurrentVariableEnforce doConcurrentVariableEnforce{
          context_, currentStatementSourcePosition_};
      parser::Walk(block, doConcurrentVariableEnforce);
    }
  }

  // C1123, concurrent limit or step expressions can't reference index-names
  void CheckConcurrentHeader(const parser::ConcurrentHeader &header) const {
    if (const auto &mask{
            std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)}) {
      CheckMaskIsPure(*mask);
    }
    auto &controls{std::get<std::list<parser::ConcurrentControl>>(header.t)};
    UnorderedSymbolSet indexNames;
    for (const parser::ConcurrentControl &control : controls) {
      const auto &indexName{std::get<parser::Name>(control.t)};
      if (indexName.symbol) {
        indexNames.insert(*indexName.symbol);
      }
    }
    if (!indexNames.empty()) {
      for (const parser::ConcurrentControl &control : controls) {
        HasNoReferences(indexNames, std::get<1>(control.t));
        HasNoReferences(indexNames, std::get<2>(control.t));
        if (const auto &intExpr{
                std::get<std::optional<parser::ScalarIntExpr>>(control.t)}) {
          const parser::Expr &expr{intExpr->thing.thing.value()};
          CheckNoCollisions(GatherSymbolsFromExpression(expr), indexNames,
              "%s step expression may not reference index variable '%s'"_err_en_US,
              expr.source);
          if (IsZero(expr)) {
            context_.Say(expr.source,
                "%s step expression may not be zero"_err_en_US, LoopKindName());
          }
        }
      }
    }
  }

  void CheckLocalitySpecs(
      const parser::LoopControl &control, const parser::Block &block) const {
    const auto &concurrent{
        std::get<parser::LoopControl::Concurrent>(control.u)};
    const auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
    const auto &localitySpecs{
        std::get<std::list<parser::LocalitySpec>>(concurrent.t)};
    if (!localitySpecs.empty()) {
      const UnorderedSymbolSet &localVars{GatherLocals(localitySpecs)};
      for (const auto &c : GetControls(control)) {
        CheckExprDoesNotReferenceLocal(std::get<1>(c.t), localVars);
        CheckExprDoesNotReferenceLocal(std::get<2>(c.t), localVars);
        if (const auto &expr{
                std::get<std::optional<parser::ScalarIntExpr>>(c.t)}) {
          CheckExprDoesNotReferenceLocal(*expr, localVars);
        }
      }
      if (const auto &mask{
              std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)}) {
        CheckMaskDoesNotReferenceLocal(*mask, localVars);
      }
      CheckDefaultNoneImpliesExplicitLocality(localitySpecs, block);
    }
  }

  // check constraints [C1121 .. C1130]
  void CheckConcurrentLoopControl(const parser::LoopControl &control) const {
    const auto &concurrent{
        std::get<parser::LoopControl::Concurrent>(control.u)};
    CheckConcurrentHeader(std::get<parser::ConcurrentHeader>(concurrent.t));
  }

  template <typename T> void CheckForImpureCall(const T &x) {
    if (auto bad{FindImpureCall(context_.foldingContext(), x)}) {
      context_.Say(
          "Impure procedure '%s' may not be referenced in a %s"_err_en_US, *bad,
          LoopKindName());
    }
  }

  // Each index should be used on the LHS of each assignment in a FORALL
  void CheckForallIndexesUsed(const evaluate::Assignment &assignment) {
    SymbolVector indexVars{context_.GetIndexVars(IndexVarKind::FORALL)};
    if (!indexVars.empty()) {
      UnorderedSymbolSet symbols{evaluate::CollectSymbols(assignment.lhs)};
      std::visit(
          common::visitors{
              [&](const evaluate::Assignment::BoundsSpec &spec) {
                for (const auto &bound : spec) {
// TODO: this is working around missing std::set::merge in some versions of
// clang that we are building with
#ifdef __clang__
                  auto boundSymbols{evaluate::CollectSymbols(bound)};
                  symbols.insert(boundSymbols.begin(), boundSymbols.end());
#else
                  symbols.merge(evaluate::CollectSymbols(bound));
#endif
                }
              },
              [&](const evaluate::Assignment::BoundsRemapping &remapping) {
                for (const auto &bounds : remapping) {
#ifdef __clang__
                  auto lbSymbols{evaluate::CollectSymbols(bounds.first)};
                  symbols.insert(lbSymbols.begin(), lbSymbols.end());
                  auto ubSymbols{evaluate::CollectSymbols(bounds.second)};
                  symbols.insert(ubSymbols.begin(), ubSymbols.end());
#else
                  symbols.merge(evaluate::CollectSymbols(bounds.first));
                  symbols.merge(evaluate::CollectSymbols(bounds.second));
#endif
                }
              },
              [](const auto &) {},
          },
          assignment.u);
      for (const Symbol &index : indexVars) {
        if (symbols.count(index) == 0) {
          context_.Say("FORALL index variable '%s' not used on left-hand side"
                       " of assignment"_warn_en_US,
              index.name());
        }
      }
    }
  }

  // For messages where the DO loop must be DO CONCURRENT, make that explicit.
  const char *LoopKindName() const {
    return kind_ == IndexVarKind::DO ? "DO CONCURRENT" : "FORALL";
  }

  SemanticsContext &context_;
  const IndexVarKind kind_;
  parser::CharBlock currentStatementSourcePosition_;
}; // class DoContext

void DoForallChecker::Enter(const parser::DoConstruct &doConstruct) {
  DoContext doContext{context_, IndexVarKind::DO};
  doContext.DefineDoVariables(doConstruct);
}

void DoForallChecker::Leave(const parser::DoConstruct &doConstruct) {
  DoContext doContext{context_, IndexVarKind::DO};
  doContext.Check(doConstruct);
  doContext.ResetDoVariables(doConstruct);
}

void DoForallChecker::Enter(const parser::ForallConstruct &construct) {
  DoContext doContext{context_, IndexVarKind::FORALL};
  doContext.ActivateIndexVars(GetControls(construct));
}
void DoForallChecker::Leave(const parser::ForallConstruct &construct) {
  DoContext doContext{context_, IndexVarKind::FORALL};
  doContext.Check(construct);
  doContext.DeactivateIndexVars(GetControls(construct));
}

void DoForallChecker::Enter(const parser::ForallStmt &stmt) {
  DoContext doContext{context_, IndexVarKind::FORALL};
  doContext.ActivateIndexVars(GetControls(stmt));
}
void DoForallChecker::Leave(const parser::ForallStmt &stmt) {
  DoContext doContext{context_, IndexVarKind::FORALL};
  doContext.Check(stmt);
  doContext.DeactivateIndexVars(GetControls(stmt));
}
void DoForallChecker::Leave(const parser::ForallAssignmentStmt &stmt) {
  DoContext doContext{context_, IndexVarKind::FORALL};
  doContext.Check(stmt);
}

template <typename A>
static parser::CharBlock GetConstructPosition(const A &a) {
  return std::get<0>(a.t).source;
}

static parser::CharBlock GetNodePosition(const ConstructNode &construct) {
  return std::visit(
      [&](const auto &x) { return GetConstructPosition(*x); }, construct);
}

void DoForallChecker::SayBadLeave(StmtType stmtType,
    const char *enclosingStmtName, const ConstructNode &construct) const {
  context_
      .Say("%s must not leave a %s statement"_err_en_US, EnumToString(stmtType),
          enclosingStmtName)
      .Attach(GetNodePosition(construct), "The construct that was left"_en_US);
}

static const parser::DoConstruct *MaybeGetDoConstruct(
    const ConstructNode &construct) {
  if (const auto *doNode{
          std::get_if<const parser::DoConstruct *>(&construct)}) {
    return *doNode;
  } else {
    return nullptr;
  }
}

static bool ConstructIsDoConcurrent(const ConstructNode &construct) {
  const parser::DoConstruct *doConstruct{MaybeGetDoConstruct(construct)};
  return doConstruct && doConstruct->IsDoConcurrent();
}

// Check that CYCLE and EXIT statements do not cause flow of control to
// leave DO CONCURRENT, CRITICAL, or CHANGE TEAM constructs.
void DoForallChecker::CheckForBadLeave(
    StmtType stmtType, const ConstructNode &construct) const {
  std::visit(common::visitors{
                 [&](const parser::DoConstruct *doConstructPtr) {
                   if (doConstructPtr->IsDoConcurrent()) {
                     // C1135 and C1167 -- CYCLE and EXIT statements can't leave
                     // a DO CONCURRENT
                     SayBadLeave(stmtType, "DO CONCURRENT", construct);
                   }
                 },
                 [&](const parser::CriticalConstruct *) {
                   // C1135 and C1168 -- similarly, for CRITICAL
                   SayBadLeave(stmtType, "CRITICAL", construct);
                 },
                 [&](const parser::ChangeTeamConstruct *) {
                   // C1135 and C1168 -- similarly, for CHANGE TEAM
                   SayBadLeave(stmtType, "CHANGE TEAM", construct);
                 },
                 [](const auto *) {},
             },
      construct);
}

static bool StmtMatchesConstruct(const parser::Name *stmtName,
    StmtType stmtType, const std::optional<parser::Name> &constructName,
    const ConstructNode &construct) {
  bool inDoConstruct{MaybeGetDoConstruct(construct) != nullptr};
  if (!stmtName) {
    return inDoConstruct; // Unlabeled statements match all DO constructs
  } else if (constructName && constructName->source == stmtName->source) {
    return stmtType == StmtType::EXIT || inDoConstruct;
  } else {
    return false;
  }
}

// C1167 Can't EXIT from a DO CONCURRENT
void DoForallChecker::CheckDoConcurrentExit(
    StmtType stmtType, const ConstructNode &construct) const {
  if (stmtType == StmtType::EXIT && ConstructIsDoConcurrent(construct)) {
    SayBadLeave(StmtType::EXIT, "DO CONCURRENT", construct);
  }
}

// Check nesting violations for a CYCLE or EXIT statement.  Loop up the
// nesting levels looking for a construct that matches the CYCLE or EXIT
// statment.  At every construct, check for a violation.  If we find a match
// without finding a violation, the check is complete.
void DoForallChecker::CheckNesting(
    StmtType stmtType, const parser::Name *stmtName) const {
  const ConstructStack &stack{context_.constructStack()};
  for (auto iter{stack.cend()}; iter-- != stack.cbegin();) {
    const ConstructNode &construct{*iter};
    const std::optional<parser::Name> &constructName{
        MaybeGetNodeName(construct)};
    if (StmtMatchesConstruct(stmtName, stmtType, constructName, construct)) {
      CheckDoConcurrentExit(stmtType, construct);
      return; // We got a match, so we're finished checking
    }
    CheckForBadLeave(stmtType, construct);
  }

  // We haven't found a match in the enclosing constructs
  if (stmtType == StmtType::EXIT) {
    context_.Say("No matching construct for EXIT statement"_err_en_US);
  } else {
    context_.Say("No matching DO construct for CYCLE statement"_err_en_US);
  }
}

// C1135 -- Nesting for CYCLE statements
void DoForallChecker::Enter(const parser::CycleStmt &cycleStmt) {
  CheckNesting(StmtType::CYCLE, common::GetPtrFromOptional(cycleStmt.v));
}

// C1167 and C1168 -- Nesting for EXIT statements
void DoForallChecker::Enter(const parser::ExitStmt &exitStmt) {
  CheckNesting(StmtType::EXIT, common::GetPtrFromOptional(exitStmt.v));
}

void DoForallChecker::Leave(const parser::AssignmentStmt &stmt) {
  const auto &variable{std::get<parser::Variable>(stmt.t)};
  context_.CheckIndexVarRedefine(variable);
}

static void CheckIfArgIsDoVar(const evaluate::ActualArgument &arg,
    const parser::CharBlock location, SemanticsContext &context) {
  common::Intent intent{arg.dummyIntent()};
  if (intent == common::Intent::Out || intent == common::Intent::InOut) {
    if (const SomeExpr * argExpr{arg.UnwrapExpr()}) {
      if (const Symbol * var{evaluate::UnwrapWholeSymbolDataRef(*argExpr)}) {
        if (intent == common::Intent::Out) {
          context.CheckIndexVarRedefine(location, *var);
        } else {
          context.WarnIndexVarRedefine(location, *var); // INTENT(INOUT)
        }
      }
    }
  }
}

// Check to see if a DO variable is being passed as an actual argument to a
// dummy argument whose intent is OUT or INOUT.  To do this, we need to find
// the expressions for actual arguments which contain DO variables.  We get the
// intents of the dummy arguments from the ProcedureRef in the "typedCall"
// field of the CallStmt which was filled in during expression checking.  At
// the same time, we need to iterate over the parser::Expr versions of the
// actual arguments to get their source locations of the arguments for the
// messages.
void DoForallChecker::Leave(const parser::CallStmt &callStmt) {
  if (const auto &typedCall{callStmt.typedCall}) {
    const auto &parsedArgs{
        std::get<std::list<parser::ActualArgSpec>>(callStmt.v.t)};
    auto parsedArgIter{parsedArgs.begin()};
    const evaluate::ActualArguments &checkedArgs{typedCall->arguments()};
    for (const auto &checkedOptionalArg : checkedArgs) {
      if (parsedArgIter == parsedArgs.end()) {
        break; // No more parsed arguments, we're done.
      }
      const auto &parsedArg{std::get<parser::ActualArg>(parsedArgIter->t)};
      ++parsedArgIter;
      if (checkedOptionalArg) {
        const evaluate::ActualArgument &checkedArg{*checkedOptionalArg};
        if (const auto *parsedExpr{
                std::get_if<common::Indirection<parser::Expr>>(&parsedArg.u)}) {
          CheckIfArgIsDoVar(checkedArg, parsedExpr->value().source, context_);
        }
      }
    }
  }
}

void DoForallChecker::Leave(const parser::ConnectSpec &connectSpec) {
  const auto *newunit{
      std::get_if<parser::ConnectSpec::Newunit>(&connectSpec.u)};
  if (newunit) {
    context_.CheckIndexVarRedefine(newunit->v.thing.thing);
  }
}

using ActualArgumentSet = std::set<evaluate::ActualArgumentRef>;

struct CollectActualArgumentsHelper
    : public evaluate::SetTraverse<CollectActualArgumentsHelper,
          ActualArgumentSet> {
  using Base = SetTraverse<CollectActualArgumentsHelper, ActualArgumentSet>;
  CollectActualArgumentsHelper() : Base{*this} {}
  using Base::operator();
  ActualArgumentSet operator()(const evaluate::ActualArgument &arg) const {
    return Combine(ActualArgumentSet{arg},
        CollectActualArgumentsHelper{}(arg.UnwrapExpr()));
  }
};

template <typename A> ActualArgumentSet CollectActualArguments(const A &x) {
  return CollectActualArgumentsHelper{}(x);
}

template ActualArgumentSet CollectActualArguments(const SomeExpr &);

void DoForallChecker::Enter(const parser::Expr &parsedExpr) { ++exprDepth_; }

void DoForallChecker::Leave(const parser::Expr &parsedExpr) {
  CHECK(exprDepth_ > 0);
  if (--exprDepth_ == 0) { // Only check top level expressions
    if (const SomeExpr * expr{GetExpr(parsedExpr)}) {
      ActualArgumentSet argSet{CollectActualArguments(*expr)};
      for (const evaluate::ActualArgumentRef &argRef : argSet) {
        CheckIfArgIsDoVar(*argRef, parsedExpr.source, context_);
      }
    }
  }
}

void DoForallChecker::Leave(const parser::InquireSpec &inquireSpec) {
  const auto *intVar{std::get_if<parser::InquireSpec::IntVar>(&inquireSpec.u)};
  if (intVar) {
    const auto &scalar{std::get<parser::ScalarIntVariable>(intVar->t)};
    context_.CheckIndexVarRedefine(scalar.thing.thing);
  }
}

void DoForallChecker::Leave(const parser::IoControlSpec &ioControlSpec) {
  const auto *size{std::get_if<parser::IoControlSpec::Size>(&ioControlSpec.u)};
  if (size) {
    context_.CheckIndexVarRedefine(size->v.thing.thing);
  }
}

void DoForallChecker::Leave(const parser::OutputImpliedDo &outputImpliedDo) {
  const auto &control{std::get<parser::IoImpliedDoControl>(outputImpliedDo.t)};
  const parser::Name &name{control.name.thing.thing};
  context_.CheckIndexVarRedefine(name.source, *name.symbol);
}

void DoForallChecker::Leave(const parser::StatVariable &statVariable) {
  context_.CheckIndexVarRedefine(statVariable.v.thing.thing);
}

} // namespace Fortran::semantics
