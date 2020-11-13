//===-- lib/Semantics/check-acc-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "check-acc-structure.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void AccStructureChecker::Enter(const parser::AccClause::X &) { \
    CheckAllowed(llvm::acc::Clause::Y); \
  }

#define CHECK_REQ_SCALAR_INT_CONSTANT_CLAUSE(X, Y) \
  void AccStructureChecker::Enter(const parser::AccClause::X &c) { \
    CheckAllowed(llvm::acc::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::acc::Clause::Y, c.v); \
  }

namespace Fortran::semantics {

static constexpr inline AccClauseSet
    parallelAndKernelsOnlyAllowedAfterDeviceTypeClauses{
        llvm::acc::Clause::ACCC_async, llvm::acc::Clause::ACCC_wait,
        llvm::acc::Clause::ACCC_num_gangs, llvm::acc::Clause::ACCC_num_workers,
        llvm::acc::Clause::ACCC_vector_length};

static constexpr inline AccClauseSet serialOnlyAllowedAfterDeviceTypeClauses{
    llvm::acc::Clause::ACCC_async, llvm::acc::Clause::ACCC_wait};

static constexpr inline AccClauseSet loopOnlyAllowedAfterDeviceTypeClauses{
    llvm::acc::Clause::ACCC_auto, llvm::acc::Clause::ACCC_collapse,
    llvm::acc::Clause::ACCC_independent, llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_seq, llvm::acc::Clause::ACCC_tile,
    llvm::acc::Clause::ACCC_vector, llvm::acc::Clause::ACCC_worker};

static constexpr inline AccClauseSet updateOnlyAllowedAfterDeviceTypeClauses{
    llvm::acc::Clause::ACCC_async, llvm::acc::Clause::ACCC_wait};

static constexpr inline AccClauseSet routineOnlyAllowedAfterDeviceTypeClauses{
    llvm::acc::Clause::ACCC_bind, llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_vector, llvm::acc::Clause::ACCC_worker};

bool AccStructureChecker::CheckAllowedModifier(llvm::acc::Clause clause) {
  if (GetContext().directive == llvm::acc::ACCD_enter_data ||
      GetContext().directive == llvm::acc::ACCD_exit_data) {
    context_.Say(GetContext().clauseSource,
        "Modifier is not allowed for the %s clause "
        "on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(clause).str()),
        ContextDirectiveAsFortran());
    return true;
  }
  return false;
}

void AccStructureChecker::Enter(const parser::AccClause &x) {
  SetContextClause(x);
}

void AccStructureChecker::Leave(const parser::AccClauseList &) {}

void AccStructureChecker::Enter(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::AccEndBlockDirective>(x.t)};
  const auto &beginAccBlockDir{
      std::get<parser::AccBlockDirective>(beginBlockDir.t)};

  CheckMatching(beginAccBlockDir, endBlockDir.v);
  PushContextAndClauseSets(beginAccBlockDir.source, beginAccBlockDir.v);
}

void AccStructureChecker::Leave(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &blockDir{std::get<parser::AccBlockDirective>(beginBlockDir.t)};
  const parser::Block &block{std::get<parser::Block>(x.t)};
  switch (blockDir.v) {
  case llvm::acc::Directive::ACCD_kernels:
  case llvm::acc::Directive::ACCD_parallel:
    // Restriction - 880-881 (KERNELS)
    // Restriction - 843-844 (PARALLEL)
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        parallelAndKernelsOnlyAllowedAfterDeviceTypeClauses);
    // Restriction - 877 (KERNELS)
    // Restriction - 840 (PARALLEL)
    CheckNoBranching(block, GetContext().directive, blockDir.source);
    break;
  case llvm::acc::Directive::ACCD_serial:
    // Restriction - 919
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        serialOnlyAllowedAfterDeviceTypeClauses);
    // Restriction - 916
    CheckNoBranching(block, llvm::acc::Directive::ACCD_serial, blockDir.source);
    break;
  case llvm::acc::Directive::ACCD_data:
    // Restriction - 1117-1118
    CheckRequireAtLeastOneOf();
    break;
  case llvm::acc::Directive::ACCD_host_data:
    // Restriction - 1578
    CheckRequireAtLeastOneOf();
    break;
  default:
    break;
  }
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(
    const parser::OpenACCStandaloneDeclarativeConstruct &x) {
  const auto &declarativeDir{std::get<parser::AccDeclarativeDirective>(x.t)};
  PushContextAndClauseSets(declarativeDir.source, declarativeDir.v);
}

void AccStructureChecker::Leave(
    const parser::OpenACCStandaloneDeclarativeConstruct &) {
  // Restriction - 2075
  CheckAtLeastOneClause();
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginCombinedDir{
      std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &combinedDir{
      std::get<parser::AccCombinedDirective>(beginCombinedDir.t)};

  // check matching, End directive is optional
  if (const auto &endCombinedDir{
          std::get<std::optional<parser::AccEndCombinedDirective>>(x.t)}) {
    CheckMatching<parser::AccCombinedDirective>(combinedDir, endCombinedDir->v);
  }

  PushContextAndClauseSets(combinedDir.source, combinedDir.v);
}

void AccStructureChecker::Leave(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &combinedDir{
      std::get<parser::AccCombinedDirective>(beginBlockDir.t)};
  switch (combinedDir.v) {
  case llvm::acc::Directive::ACCD_kernels_loop:
  case llvm::acc::Directive::ACCD_parallel_loop:
    // Restriction - 1962 -> (880-881) (KERNELS LOOP)
    // Restriction - 1962 -> (843-844) (PARALLEL LOOP)
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        {llvm::acc::Clause::ACCC_async, llvm::acc::Clause::ACCC_wait,
            llvm::acc::Clause::ACCC_num_gangs,
            llvm::acc::Clause::ACCC_num_workers,
            llvm::acc::Clause::ACCC_vector_length});
    break;
  case llvm::acc::Directive::ACCD_serial_loop:
    // Restriction - 1962 -> (919) (SERIAL LOOP)
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        {llvm::acc::Clause::ACCC_async, llvm::acc::Clause::ACCC_wait});
    break;
  default:
    break;
  }
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCLoopConstruct &x) {
  const auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
  const auto &loopDir{std::get<parser::AccLoopDirective>(beginDir.t)};
  PushContextAndClauseSets(loopDir.source, loopDir.v);
}

void AccStructureChecker::Leave(const parser::OpenACCLoopConstruct &x) {
  const auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
  const auto &loopDir{std::get<parser::AccLoopDirective>(beginDir.t)};
  if (loopDir.v == llvm::acc::Directive::ACCD_loop) {
    // Restriction - 1615-1616
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        loopOnlyAllowedAfterDeviceTypeClauses);
    // Restriction - 1622
    CheckNotAllowedIfClause(llvm::acc::Clause::ACCC_seq,
        {llvm::acc::Clause::ACCC_gang, llvm::acc::Clause::ACCC_vector,
            llvm::acc::Clause::ACCC_worker});
  }
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::AccStandaloneDirective>(x.t)};
  PushContextAndClauseSets(standaloneDir.source, standaloneDir.v);
}

void AccStructureChecker::Leave(const parser::OpenACCStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::AccStandaloneDirective>(x.t)};
  switch (standaloneDir.v) {
  case llvm::acc::Directive::ACCD_enter_data:
  case llvm::acc::Directive::ACCD_exit_data:
  case llvm::acc::Directive::ACCD_set:
    // Restriction - 1117-1118 (ENTER DATA)
    // Restriction - 1161-1162 (EXIT DATA)
    // Restriction - 2254 (SET)
    CheckRequireAtLeastOneOf();
    break;
  case llvm::acc::Directive::ACCD_update:
    // Restriction - 2301
    CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
        updateOnlyAllowedAfterDeviceTypeClauses);
    break;
  default:
    break;
  }
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCRoutineConstruct &x) {
  PushContextAndClauseSets(x.source, llvm::acc::Directive::ACCD_routine);
}
void AccStructureChecker::Leave(const parser::OpenACCRoutineConstruct &) {
  // Restriction - 2409
  CheckRequireAtLeastOneOf();
  // Restriction - 2407-2408
  CheckOnlyAllowedAfter(llvm::acc::Clause::ACCC_device_type,
      routineOnlyAllowedAfterDeviceTypeClauses);
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCWaitConstruct &x) {
  const auto &verbatim{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(verbatim.source, llvm::acc::Directive::ACCD_wait);
}
void AccStructureChecker::Leave(const parser::OpenACCWaitConstruct &x) {
  dirContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCAtomicConstruct &x) {
  PushContextAndClauseSets(x.source, llvm::acc::Directive::ACCD_atomic);
}
void AccStructureChecker::Leave(const parser::OpenACCAtomicConstruct &x) {
  dirContext_.pop_back();
}

// Clause checkers
CHECK_REQ_SCALAR_INT_CONSTANT_CLAUSE(Collapse, ACCC_collapse)

CHECK_SIMPLE_CLAUSE(Auto, ACCC_auto)
CHECK_SIMPLE_CLAUSE(Async, ACCC_async)
CHECK_SIMPLE_CLAUSE(Attach, ACCC_attach)
CHECK_SIMPLE_CLAUSE(Bind, ACCC_bind)
CHECK_SIMPLE_CLAUSE(Capture, ACCC_capture)
CHECK_SIMPLE_CLAUSE(Copy, ACCC_copy)
CHECK_SIMPLE_CLAUSE(Default, ACCC_default)
CHECK_SIMPLE_CLAUSE(DefaultAsync, ACCC_default_async)
CHECK_SIMPLE_CLAUSE(Delete, ACCC_delete)
CHECK_SIMPLE_CLAUSE(Detach, ACCC_detach)
CHECK_SIMPLE_CLAUSE(Device, ACCC_device)
CHECK_SIMPLE_CLAUSE(DeviceNum, ACCC_device_num)
CHECK_SIMPLE_CLAUSE(Deviceptr, ACCC_deviceptr)
CHECK_SIMPLE_CLAUSE(DeviceResident, ACCC_device_resident)
CHECK_SIMPLE_CLAUSE(DeviceType, ACCC_device_type)
CHECK_SIMPLE_CLAUSE(Finalize, ACCC_finalize)
CHECK_SIMPLE_CLAUSE(Firstprivate, ACCC_firstprivate)
CHECK_SIMPLE_CLAUSE(Gang, ACCC_gang)
CHECK_SIMPLE_CLAUSE(Host, ACCC_host)
CHECK_SIMPLE_CLAUSE(If, ACCC_if)
CHECK_SIMPLE_CLAUSE(IfPresent, ACCC_if_present)
CHECK_SIMPLE_CLAUSE(Independent, ACCC_independent)
CHECK_SIMPLE_CLAUSE(Link, ACCC_link)
CHECK_SIMPLE_CLAUSE(NoCreate, ACCC_no_create)
CHECK_SIMPLE_CLAUSE(Nohost, ACCC_nohost)
CHECK_SIMPLE_CLAUSE(NumGangs, ACCC_num_gangs)
CHECK_SIMPLE_CLAUSE(NumWorkers, ACCC_num_workers)
CHECK_SIMPLE_CLAUSE(Present, ACCC_present)
CHECK_SIMPLE_CLAUSE(Private, ACCC_private)
CHECK_SIMPLE_CLAUSE(Read, ACCC_read)
CHECK_SIMPLE_CLAUSE(Reduction, ACCC_reduction)
CHECK_SIMPLE_CLAUSE(Self, ACCC_self)
CHECK_SIMPLE_CLAUSE(Seq, ACCC_seq)
CHECK_SIMPLE_CLAUSE(Tile, ACCC_tile)
CHECK_SIMPLE_CLAUSE(UseDevice, ACCC_use_device)
CHECK_SIMPLE_CLAUSE(Vector, ACCC_vector)
CHECK_SIMPLE_CLAUSE(VectorLength, ACCC_vector_length)
CHECK_SIMPLE_CLAUSE(Wait, ACCC_wait)
CHECK_SIMPLE_CLAUSE(Worker, ACCC_worker)
CHECK_SIMPLE_CLAUSE(Write, ACCC_write)

void AccStructureChecker::Enter(const parser::AccClause::Create &c) {
  CheckAllowed(llvm::acc::Clause::ACCC_create);
  const auto &modifierClause{c.v};
  if (const auto &modifier{
          std::get<std::optional<parser::AccDataModifier>>(modifierClause.t)}) {
    if (modifier->v != parser::AccDataModifier::Modifier::Zero) {
      context_.Say(GetContext().clauseSource,
          "Only the ZERO modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::acc::getOpenACCClauseName(llvm::acc::Clause::ACCC_create)
                  .str()),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::Enter(const parser::AccClause::Copyin &c) {
  CheckAllowed(llvm::acc::Clause::ACCC_copyin);
  const auto &modifierClause{c.v};
  if (const auto &modifier{
          std::get<std::optional<parser::AccDataModifier>>(modifierClause.t)}) {
    if (CheckAllowedModifier(llvm::acc::Clause::ACCC_copyin))
      return;
    if (modifier->v != parser::AccDataModifier::Modifier::ReadOnly) {
      context_.Say(GetContext().clauseSource,
          "Only the READONLY modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::acc::getOpenACCClauseName(llvm::acc::Clause::ACCC_copyin)
                  .str()),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::Enter(const parser::AccClause::Copyout &c) {
  CheckAllowed(llvm::acc::Clause::ACCC_copyout);
  const auto &modifierClause{c.v};
  if (const auto &modifier{
          std::get<std::optional<parser::AccDataModifier>>(modifierClause.t)}) {
    if (CheckAllowedModifier(llvm::acc::Clause::ACCC_copyout))
      return;
    if (modifier->v != parser::AccDataModifier::Modifier::Zero) {
      context_.Say(GetContext().clauseSource,
          "Only the ZERO modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::acc::getOpenACCClauseName(llvm::acc::Clause::ACCC_copyout)
                  .str()),
          ContextDirectiveAsFortran());
    }
  }
}

llvm::StringRef AccStructureChecker::getClauseName(llvm::acc::Clause clause) {
  return llvm::acc::getOpenACCClauseName(clause);
}

llvm::StringRef AccStructureChecker::getDirectiveName(
    llvm::acc::Directive directive) {
  return llvm::acc::getOpenACCDirectiveName(directive);
}

} // namespace Fortran::semantics
