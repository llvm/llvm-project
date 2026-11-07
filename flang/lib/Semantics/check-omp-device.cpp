//===-- lib/Semantics/check-omp-device.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-device.h"

#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

#include <deque>
#include <string>

namespace Fortran::semantics {

// Collects the procedures called in an expression. Procedures passed as
// actual arguments are not called by the expression, so they are skipped.
struct CalleeCollector
    : public evaluate::Traverse<CalleeCollector, std::vector<const Symbol *>> {
  using Result = std::vector<const Symbol *>;
  using Base = evaluate::Traverse<CalleeCollector, Result>;
  using Base::operator();
  CalleeCollector() : Base{*this} {}
  Result Default() { return {}; }
  Result Combine(Result &&a, Result &&b) {
    a.insert(a.end(), b.begin(), b.end());
    return std::move(a);
  }
  Result operator()(const evaluate::ProcedureDesignator &) const { return {}; }
  Result operator()(const evaluate::ProcedureRef &x) const {
    Result result{(*this)(x.arguments())};
    if (const Symbol *callee{x.proc().GetSymbol()}) {
      result.push_back(callee);
    }
    return result;
  }
};

static bool IsThreadprivate(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<CommonBlockDetails>()}) {
    return llvm::any_of(details->objects(), [](const MutableSymbolRef &object) {
      return object->GetUltimate().test(Symbol::Flag::OmpThreadprivate);
    });
  }
  return symbol.test(Symbol::Flag::OmpThreadprivate);
}

static std::string GetVariableName(const Symbol &symbol) {
  if (symbol.has<CommonBlockDetails>()) {
    return "/" + symbol.name().ToString() + "/";
  }
  return symbol.name().ToString();
}

OmpDeviceChecker::OmpDeviceChecker(SemanticsContext &context)
    : context_{context},
      isTargetDevice_{context.langOptions().OpenMPIsTargetDevice} {}

void OmpDeviceChecker::Leave(const parser::Program &) {
  if (isTargetDevice_) {
    CheckImplicitDeclareTarget();
  }
}

void OmpDeviceChecker::Enter(const parser::SubroutineSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(x.t)};
  EnterProcedure(std::get<parser::Name>(stmt.statement.t).symbol);
}

void OmpDeviceChecker::Leave(const parser::SubroutineSubprogram &) {
  LeaveProcedure();
}

void OmpDeviceChecker::Enter(const parser::FunctionSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(x.t)};
  EnterProcedure(std::get<parser::Name>(stmt.statement.t).symbol);
}

void OmpDeviceChecker::Leave(const parser::FunctionSubprogram &) {
  LeaveProcedure();
}

void OmpDeviceChecker::Enter(const parser::SeparateModuleSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::MpSubprogramStmt>>(x.t)};
  EnterProcedure(stmt.statement.v.symbol);
}

void OmpDeviceChecker::Leave(const parser::SeparateModuleSubprogram &) {
  LeaveProcedure();
}

void OmpDeviceChecker::Enter(const parser::ExecutionPart &) {
  inExecutionPart_ = true;
}

void OmpDeviceChecker::Leave(const parser::ExecutionPart &) {
  inExecutionPart_ = false;
}

void OmpDeviceChecker::Enter(const parser::EntryStmt &x) {
  const Symbol *procedure{CurrentProcedure()};
  const Symbol *entry{std::get<parser::Name>(x.t).symbol};
  if (procedure && entry) {
    aliases_[entry] = procedure;
  }
}

void OmpDeviceChecker::Leave(const parser::OmpBeginDirective &x) {
  // The body of a TARGET construct is device code, but its clauses are
  // evaluated by the encountering task.
  if (llvm::omp::allTargetSet.test(x.DirId())) {
    targets_.push_back(&x);
  }
}

void OmpDeviceChecker::Leave(const parser::OmpBlockConstruct &x) {
  LeaveConstruct(x.BeginDir());
}

void OmpDeviceChecker::Leave(const parser::OpenMPLoopConstruct &x) {
  LeaveConstruct(x.BeginDir());
}

void OmpDeviceChecker::Enter(const parser::OmpClause &x) {
  clauses_.push_back(x.Id());
}

void OmpDeviceChecker::Leave(const parser::OmpClause &) { clauses_.pop_back(); }

void OmpDeviceChecker::Enter(const parser::OmpObjectList &) {
  // THREADPRIVATE variables in clauses are diagnosed by the clause checks,
  // except in the clauses that allow them. Those clauses access them.
  bool skip{clauses_.empty() ||
      (clauses_.back() != llvm::omp::Clause::OMPC_copyin &&
          clauses_.back() != llvm::omp::Clause::OMPC_copyprivate)};
  skippedObjectLists_.push_back(skip);
  skipDepth_ += skip;
}

void OmpDeviceChecker::Leave(const parser::OmpObjectList &) {
  skipDepth_ -= skippedObjectLists_.back();
  skippedObjectLists_.pop_back();
}

// Objects in directive arguments (e.g. FLUSH) are not checked.
void OmpDeviceChecker::Enter(const parser::OmpArgumentList &) { ++skipDepth_; }

void OmpDeviceChecker::Leave(const parser::OmpArgumentList &) { --skipDepth_; }

// DATA statements initialize variables before execution starts.
void OmpDeviceChecker::Enter(const parser::DataStmt &) { ++skipDepth_; }

void OmpDeviceChecker::Leave(const parser::DataStmt &) { --skipDepth_; }

void OmpDeviceChecker::Enter(const parser::Name &x) {
  if (inExecutionPart_ && skipDepth_ == 0 && x.symbol) {
    CheckReference(x.source, *x.symbol);
  }
}

void OmpDeviceChecker::Enter(const parser::Expr &x) {
  // The analyzed outermost expression contains all calls.
  if (exprDepth_++ == 0 && inExecutionPart_ && skipDepth_ == 0) {
    if (const SomeExpr *expr{GetExpr(context_, x)}) {
      RecordCalls(x.source, *expr);
    }
  }
}

void OmpDeviceChecker::Leave(const parser::Expr &) { --exprDepth_; }

void OmpDeviceChecker::Enter(const parser::CallStmt &x) {
  if (inExecutionPart_ && x.typedCall) {
    if (const Symbol *callee{x.typedCall->proc().GetSymbol()}) {
      RecordCall(x.source, *callee);
    }
  }
}

void OmpDeviceChecker::Enter(const parser::AssignmentStmt &x) {
  // Defined assignment
  if (inExecutionPart_) {
    if (const auto *assignment{GetAssignment(x)}) {
      if (const auto *call{
              std::get_if<evaluate::ProcedureRef>(&assignment->u)}) {
        if (const Symbol *callee{call->proc().GetSymbol()}) {
          RecordCall(std::get<parser::Variable>(x.t).GetSource(), *callee);
        }
      }
    }
  }
}

void OmpDeviceChecker::EnterProcedure(const Symbol *symbol) {
  procedures_.push_back(symbol);
  if (!symbol) {
    return;
  }
  ProcedureInfo &info{procedureInfo_[symbol]};
  const auto *details{symbol->detailsIf<SubprogramDetails>()};
  if (!details) {
    return;
  }
  // DECLARE TARGET is recorded in the details of the procedure, or of the
  // interface of a separate module procedure. (The OmpDeclareTarget flag may
  // be set on a host-associated symbol of the procedure instead.)
  const SubprogramDetails *declaration{details};
  if (const Symbol *interface{details->moduleInterface()}) {
    // Calls of a separate module procedure refer to its interface.
    const Symbol &ultimate{interface->GetUltimate()};
    aliases_[&ultimate] = symbol;
    if (details->ompDeclTarget().empty()) {
      if (const auto *interfaceDetails{
              ultimate.detailsIf<SubprogramDetails>()}) {
        declaration = interfaceDetails;
      }
    }
  }
  if (!declaration->ompDeclTarget().empty()) {
    info.isExplicitDeclareTarget = true;
    const auto &deviceType{declaration->ompDeclTargetDeviceType()};
    info.isExplicitDevice =
        !deviceType || *deviceType != common::OmpDeviceType::Host;
  }
}

void OmpDeviceChecker::LeaveProcedure() { procedures_.pop_back(); }

void OmpDeviceChecker::LeaveConstruct(const parser::OmpBeginDirective &x) {
  if (!targets_.empty() && targets_.back() == &x) {
    targets_.pop_back();
  }
}

OmpDeviceChecker::DeviceContext OmpDeviceChecker::GetDeviceContext() const {
  if (!targets_.empty()) {
    return DeviceContext::TargetRegion;
  }
  if (const Symbol *procedure{CurrentProcedure()};
      procedure && inExecutionPart_) {
    if (auto iter{procedureInfo_.find(procedure)};
        iter != procedureInfo_.end() && iter->second.isExplicitDevice) {
      return DeviceContext::DeclareTarget;
    }
  }
  return DeviceContext::None;
}

const Symbol *OmpDeviceChecker::CurrentProcedure() const {
  return procedures_.empty() ? nullptr : procedures_.back();
}

void OmpDeviceChecker::CheckReference(parser::CharBlock source,
    const Symbol &symbol, const Symbol *statementFunction) {
  // Resolve use, host, and construct associations.
  const Symbol &root{GetAssociationRoot(symbol)};
  if (const auto *details{root.detailsIf<SubprogramDetails>()}) {
    // The expression of a statement function is evaluated where the statement
    // function is referenced.
    if (const auto &expr{details->stmtFunction()};
        expr && expanding_.insert(&root).second) {
      const Symbol *outermost{statementFunction ? statementFunction : &root};
      for (const Symbol &referenced : evaluate::CollectSymbols(*expr)) {
        CheckReference(source, referenced, outermost);
      }
      RecordCalls(source, *expr);
      expanding_.erase(&root);
    }
  } else if (IsThreadprivate(root)) {
    CheckAccess(Access{source, &root, statementFunction});
  }
}

void OmpDeviceChecker::CheckAccess(const Access &access) {
  DeviceContext context{GetDeviceContext()};
  if (context == DeviceContext::None) {
    // Diagnosed later if the procedure turns out to be device code.
    if (const Symbol *procedure{CurrentProcedure()};
        procedure && isTargetDevice_) {
      procedureInfo_[procedure].accesses.push_back(access);
    }
    return;
  }
  // OpenMP 5.2: 5.2 threadprivate directive restriction: the effect of an
  // access to a threadprivate variable in a target region is unspecified.
  std::string name{GetVariableName(*access.variable)};
  auto &message{context_.Say(access.source,
      "A THREADPRIVATE variable '%s' cannot be accessed in a %s"_err_en_US,
      name,
      context == DeviceContext::TargetRegion ? "TARGET region"
                                             : "DECLARE TARGET procedure")};
  if (access.statementFunction) {
    message.Attach(access.statementFunction->name(),
        "'%s' is accessed by statement function '%s'"_en_US, name,
        access.statementFunction->name());
  }
}

void OmpDeviceChecker::RecordCall(
    parser::CharBlock source, const Symbol &callee) {
  if (!isTargetDevice_) {
    return;
  }
  Call call{source, &callee, CurrentProcedure(), GetDeviceContext()};
  if (call.context != DeviceContext::None) {
    deviceCalls_.push_back(call);
  } else if (call.caller) {
    procedureInfo_[call.caller].calls.push_back(call);
  }
}

void OmpDeviceChecker::RecordCalls(
    parser::CharBlock source, const SomeExpr &expr) {
  for (const Symbol *callee : CalleeCollector{}(expr)) {
    RecordCall(source, *callee);
  }
}

// Returns the definition in the compilation unit of the called procedure.
const Symbol *OmpDeviceChecker::FindDefinition(const Symbol &callee) const {
  auto lookup{[&](const Symbol *symbol) -> const Symbol * {
    if (auto iter{aliases_.find(symbol)}; iter != aliases_.end()) {
      symbol = iter->second;
    }
    return procedureInfo_.count(symbol) ? symbol : nullptr;
  }};
  const Symbol *symbol{&callee.GetUltimate()};
  if (const auto *binding{symbol->detailsIf<ProcBindingDetails>()}) {
    symbol = &binding->symbol().GetUltimate();
  }
  if (const Symbol *definition{lookup(symbol)}) {
    return definition;
  }
  // An external procedure called through an interface.
  if (const Symbol *global{FindGlobal(*symbol)}) {
    return lookup(global);
  }
  return nullptr;
}

// A procedure that is called from device code and is not explicitly DECLARE
// TARGET is implicitly DECLARE TARGET for the device. Propagate this along the
// calls, like the implicit DECLARE TARGET marking during lowering, and check
// the accesses of the reached procedures.
void OmpDeviceChecker::CheckImplicitDeclareTarget() {
  // The call through which each procedure was first reached.
  llvm::DenseMap<const Symbol *, Call> reachedBy;
  std::deque<Call> worklist(deviceCalls_.begin(), deviceCalls_.end());
  while (!worklist.empty()) {
    Call call{worklist.front()};
    worklist.pop_front();
    const Symbol *procedure{FindDefinition(*call.callee)};
    if (!procedure) {
      continue;
    }
    const ProcedureInfo &info{procedureInfo_.find(procedure)->second};
    if (info.isExplicitDeclareTarget ||
        !reachedBy.try_emplace(procedure, call).second) {
      continue;
    }
    for (const Access &access : info.accesses) {
      std::string name{GetVariableName(*access.variable)};
      auto &message{context_.Say(access.source,
          "A THREADPRIVATE variable '%s' cannot be accessed in procedure '%s', which is implicitly DECLARE TARGET"_err_en_US,
          name, procedure->name())};
      if (access.statementFunction) {
        message.Attach(access.statementFunction->name(),
            "'%s' is accessed by statement function '%s'"_en_US, name,
            access.statementFunction->name());
      }
      // Explain how the procedure is called from device code.
      for (const Symbol *callee{procedure};;) {
        const Call &edge{reachedBy.find(callee)->second};
        if (edge.context == DeviceContext::TargetRegion) {
          message.Attach(edge.source,
              "'%s' is called from a TARGET region here"_en_US, callee->name());
          break;
        }
        if (edge.context == DeviceContext::DeclareTarget) {
          message.Attach(edge.source,
              "'%s' is called from DECLARE TARGET procedure '%s' here"_en_US,
              callee->name(), edge.caller->name());
          break;
        }
        message.Attach(edge.source, "'%s' is called from '%s' here"_en_US,
            callee->name(), edge.caller->name());
        callee = edge.caller;
      }
    }
    worklist.insert(worklist.end(), info.calls.begin(), info.calls.end());
  }
}

} // namespace Fortran::semantics
