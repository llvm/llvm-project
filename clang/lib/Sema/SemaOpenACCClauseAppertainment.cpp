//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements conversions from the ACC.td from the backend to
/// determine appertainment, required/etc.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/SemaOpenACC.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

namespace {
// Implements a simple 'enum-set' which stores enum values in a single 64 bit
// value. Flang has `EnumSet` which is pretty sizable/has a lot of dependencies,
// so likely not worth bringing in for this use.
class AccClauseSet {
  // We're just using a uint64_t as our underlying rep, so if this size ever
  // gets bigger than 64, we probably need a pair of uint64_ts.
  static_assert(llvm::acc::Clause_enumSize <= 64);
  uint64_t Data;

  void setBit(llvm::acc::Clause C) {
    Data |= static_cast<uint64_t>(1) << static_cast<uint64_t>(C);
  }

public:
  constexpr AccClauseSet(
      const std::initializer_list<llvm::acc::Clause> &Clauses)
      : Data(0) {
    for (llvm::acc::Clause C : Clauses)
      setBit(C);
  }

  constexpr bool isSet(llvm::acc::Clause C) const {
    return ((Data >> static_cast<uint64_t>(C)) & 1) != 0;
  }

  void clearBit(llvm::acc::Clause C) {
    Data &= ~(static_cast<uint64_t>(1) << static_cast<uint64_t>(C));
  }

  constexpr bool isEmpty() const { return Data == 0; }

  unsigned popcount() const { return llvm::popcount<uint64_t>(Data); }
};
} // namespace

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenACC/ACC.inc"

using namespace clang;

namespace {
struct LLVMClauseLists {
  AccClauseSet Allowed;
  AccClauseSet AllowedOnce;
  AccClauseSet AllowedExclusive;
  AccClauseSet Required;
};
struct LLVMDirectiveClauseRelationships {
  llvm::acc::Directive DirKind;
  LLVMClauseLists Lists;
};

LLVMDirectiveClauseRelationships Relations[] =
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenACC/ACC.inc"
    ;

llvm::acc::Directive
getLLVMDirectiveFromClangDirective(OpenACCDirectiveKind DK) {
  // FIXME: There isn't any obvious way to do this automatically, but perhaps we
  // could figure out one?
  switch (DK) {
  case OpenACCDirectiveKind::Parallel:
    return llvm::acc::Directive::ACCD_parallel;
  case OpenACCDirectiveKind::Serial:
    return llvm::acc::Directive::ACCD_serial;
  case OpenACCDirectiveKind::Kernels:
    return llvm::acc::Directive::ACCD_kernels;
  case OpenACCDirectiveKind::Data:
    return llvm::acc::Directive::ACCD_data;
  case OpenACCDirectiveKind::EnterData:
    return llvm::acc::Directive::ACCD_enter_data;
  case OpenACCDirectiveKind::ExitData:
    return llvm::acc::Directive::ACCD_exit_data;
  case OpenACCDirectiveKind::HostData:
    return llvm::acc::Directive::ACCD_host_data;
  case OpenACCDirectiveKind::Loop:
    return llvm::acc::Directive::ACCD_loop;
  case OpenACCDirectiveKind::Cache:
    return llvm::acc::Directive::ACCD_cache;
  case OpenACCDirectiveKind::ParallelLoop:
    return llvm::acc::Directive::ACCD_parallel_loop;
  case OpenACCDirectiveKind::SerialLoop:
    return llvm::acc::Directive::ACCD_serial_loop;
  case OpenACCDirectiveKind::KernelsLoop:
    return llvm::acc::Directive::ACCD_kernels_loop;
  case OpenACCDirectiveKind::Atomic:
    return llvm::acc::Directive::ACCD_atomic;
  case OpenACCDirectiveKind::Declare:
    return llvm::acc::Directive::ACCD_declare;
  case OpenACCDirectiveKind::Init:
    return llvm::acc::Directive::ACCD_init;
  case OpenACCDirectiveKind::Shutdown:
    return llvm::acc::Directive::ACCD_shutdown;
  case OpenACCDirectiveKind::Set:
    return llvm::acc::Directive::ACCD_set;
  case OpenACCDirectiveKind::Update:
    return llvm::acc::Directive::ACCD_update;
  case OpenACCDirectiveKind::Wait:
    return llvm::acc::Directive::ACCD_wait;
  case OpenACCDirectiveKind::Routine:
    return llvm::acc::Directive::ACCD_routine;
  case OpenACCDirectiveKind::Invalid:
    llvm_unreachable("Shouldn't get here with an invalid directive");
  }

  llvm_unreachable("unhandled directive kind");
}

// FIXME: There isn't any obvious way to do this automatically, but perhaps we
// could figure out one?
std::pair<OpenACCClauseKind, llvm::acc::Clause> ClauseEquivs[]{
    {OpenACCClauseKind::Finalize, llvm::acc::Clause::ACCC_finalize},
    {OpenACCClauseKind::IfPresent, llvm::acc::Clause::ACCC_if_present},
    {OpenACCClauseKind::Seq, llvm::acc::Clause::ACCC_seq},
    {OpenACCClauseKind::Independent, llvm::acc::Clause::ACCC_independent},
    {OpenACCClauseKind::Auto, llvm::acc::Clause::ACCC_auto},
    {OpenACCClauseKind::Worker, llvm::acc::Clause::ACCC_worker},
    {OpenACCClauseKind::Vector, llvm::acc::Clause::ACCC_vector},
    {OpenACCClauseKind::NoHost, llvm::acc::Clause::ACCC_nohost},
    {OpenACCClauseKind::Default, llvm::acc::Clause::ACCC_default},
    {OpenACCClauseKind::If, llvm::acc::Clause::ACCC_if},
    {OpenACCClauseKind::Self, llvm::acc::Clause::ACCC_self},
    {OpenACCClauseKind::Copy, llvm::acc::Clause::ACCC_copy},
    {OpenACCClauseKind::PCopy, llvm::acc::Clause::ACCC_copy},
    {OpenACCClauseKind::PresentOrCopy, llvm::acc::Clause::ACCC_copy},
    {OpenACCClauseKind::UseDevice, llvm::acc::Clause::ACCC_use_device},
    {OpenACCClauseKind::Attach, llvm::acc::Clause::ACCC_attach},
    {OpenACCClauseKind::Delete, llvm::acc::Clause::ACCC_delete},
    {OpenACCClauseKind::Detach, llvm::acc::Clause::ACCC_detach},
    {OpenACCClauseKind::Device, llvm::acc::Clause::ACCC_device},
    {OpenACCClauseKind::DevicePtr, llvm::acc::Clause::ACCC_deviceptr},
    {OpenACCClauseKind::DeviceResident,
     llvm::acc::Clause::ACCC_device_resident},
    {OpenACCClauseKind::FirstPrivate, llvm::acc::Clause::ACCC_firstprivate},
    {OpenACCClauseKind::Host, llvm::acc::Clause::ACCC_host},
    {OpenACCClauseKind::Link, llvm::acc::Clause::ACCC_link},
    {OpenACCClauseKind::NoCreate, llvm::acc::Clause::ACCC_no_create},
    {OpenACCClauseKind::Present, llvm::acc::Clause::ACCC_present},
    {OpenACCClauseKind::Private, llvm::acc::Clause::ACCC_private},
    {OpenACCClauseKind::CopyOut, llvm::acc::Clause::ACCC_copyout},
    {OpenACCClauseKind::PCopyOut, llvm::acc::Clause::ACCC_copyout},
    {OpenACCClauseKind::PresentOrCopyOut, llvm::acc::Clause::ACCC_copyout},
    {OpenACCClauseKind::CopyIn, llvm::acc::Clause::ACCC_copyin},
    {OpenACCClauseKind::PCopyIn, llvm::acc::Clause::ACCC_copyin},
    {OpenACCClauseKind::PresentOrCopyIn, llvm::acc::Clause::ACCC_copyin},
    {OpenACCClauseKind::Create, llvm::acc::Clause::ACCC_create},
    {OpenACCClauseKind::PCreate, llvm::acc::Clause::ACCC_create},
    {OpenACCClauseKind::PresentOrCreate, llvm::acc::Clause::ACCC_create},
    {OpenACCClauseKind::Reduction, llvm::acc::Clause::ACCC_reduction},
    {OpenACCClauseKind::Collapse, llvm::acc::Clause::ACCC_collapse},
    {OpenACCClauseKind::Bind, llvm::acc::Clause::ACCC_bind},
    {OpenACCClauseKind::VectorLength, llvm::acc::Clause::ACCC_vector_length},
    {OpenACCClauseKind::NumGangs, llvm::acc::Clause::ACCC_num_gangs},
    {OpenACCClauseKind::NumWorkers, llvm::acc::Clause::ACCC_num_workers},
    {OpenACCClauseKind::DeviceNum, llvm::acc::Clause::ACCC_device_num},
    {OpenACCClauseKind::DefaultAsync, llvm::acc::Clause::ACCC_default_async},
    {OpenACCClauseKind::DeviceType, llvm::acc::Clause::ACCC_device_type},
    {OpenACCClauseKind::DType, llvm::acc::Clause::ACCC_device_type},
    {OpenACCClauseKind::Async, llvm::acc::Clause::ACCC_async},
    {OpenACCClauseKind::Tile, llvm::acc::Clause::ACCC_tile},
    {OpenACCClauseKind::Gang, llvm::acc::Clause::ACCC_gang},
    {OpenACCClauseKind::Wait, llvm::acc::Clause::ACCC_wait}};

llvm::acc::Clause getLLVMClauseFromClangClause(OpenACCClauseKind CK) {
  assert(CK != OpenACCClauseKind::Invalid);

  auto *Res =
      llvm::find_if(ClauseEquivs, [=](auto Ref) { return CK == Ref.first; });
  assert(Res && "Unhandled clause kind");

  return Res->second;
}

OpenACCClauseKind getClangClauseFromLLVMClause(llvm::acc::Clause CK) {

  auto *Res =
      llvm::find_if(ClauseEquivs, [=](auto Ref) { return CK == Ref.second; });
  assert(Res && "Unhandled clause kind");

  return Res->first;
}

const LLVMClauseLists &getListsForDirective(OpenACCDirectiveKind DK) {

  llvm::acc::Directive Dir = getLLVMDirectiveFromClangDirective(DK);
  auto Res = llvm::find_if(Relations,
                           [=](const LLVMDirectiveClauseRelationships &Rel) {
                             return Rel.DirKind == Dir;
                           });
  assert(Res != std::end(Relations) && "Unknown directive kind?");

  return Res->Lists;
}

std::string getListOfClauses(AccClauseSet Set) {
  // We could probably come up with a better way to do this smuggling, but this
  // is good enough for now.
  std::string Output;
  llvm::raw_string_ostream OS{Output};

  for (unsigned I = 0; I < llvm::acc::Clause_enumSize; ++I) {
    llvm::acc::Clause CurClause = static_cast<llvm::acc::Clause>(I);
    if (!Set.isSet(CurClause))
      continue;

    OpenACCClauseKind NewCK = getClangClauseFromLLVMClause(CurClause);
    OS << '\'' << NewCK << '\'';

    Set.clearBit(CurClause);

    if (Set.isEmpty()) {
      OS.flush();
      return OS.str();
    }

    OS << ", ";

    if (Set.popcount() == 1)
      OS << "or ";
  }
  OS.flush();
  return OS.str();
}
} // namespace

// Diagnoses if `Clauses` list doesn't have at least one of the required
// clauses.
bool SemaOpenACC::DiagnoseRequiredClauses(
    OpenACCDirectiveKind DK, SourceLocation DirectiveLoc,
    ArrayRef<const OpenACCClause *> Clauses) {
  if (DK == OpenACCDirectiveKind::Invalid)
    return false;

  const LLVMClauseLists &Lists = getListsForDirective(DK);

  if (Lists.Required.isEmpty())
    return false;

  for (auto *C : Clauses) {
    if (Lists.Required.isSet(getLLVMClauseFromClangClause(C->getClauseKind())))
      return false;
  }

  return Diag(DirectiveLoc, diag::err_acc_construct_one_clause_of)
         << DK << getListOfClauses(Lists.Required);
  return true;
}

// Diagnoses a 'CK' on a 'DK' present more than once in a clause-list when it
// isn't allowed.
bool SemaOpenACC::DiagnoseAllowedOnceClauses(
    OpenACCDirectiveKind DK, OpenACCClauseKind CK, SourceLocation ClauseLoc,
    ArrayRef<const OpenACCClause *> Clauses) {
  if (DK == OpenACCDirectiveKind::Invalid || CK == OpenACCClauseKind::Invalid)
    return false;

  const LLVMClauseLists &Lists = getListsForDirective(DK);
  llvm::acc::Clause LLVMClause = getLLVMClauseFromClangClause(CK);

  if (!Lists.AllowedOnce.isSet(LLVMClause))
    return false;

  auto Res = llvm::find_if(Clauses, [=](const OpenACCClause *C) {
    return C->getClauseKind() == CK;
  });

  if (Res == Clauses.end())
    return false;

  Diag(ClauseLoc, diag::err_acc_duplicate_clause_disallowed) << DK << CK;
  Diag((*Res)->getBeginLoc(), diag::note_acc_previous_clause_here);
  return true;
}

// Diagnoses a 'CK' on a 'DK' being added that isn't allowed to, because another
// clause in 'Clauses' already exists.
bool SemaOpenACC::DiagnoseExclusiveClauses(
    OpenACCDirectiveKind DK, OpenACCClauseKind CK, SourceLocation ClauseLoc,
    ArrayRef<const OpenACCClause *> Clauses) {
  if (DK == OpenACCDirectiveKind::Invalid || CK == OpenACCClauseKind::Invalid)
    return false;

  llvm::acc::Clause LLVMClause = getLLVMClauseFromClangClause(CK);
  const LLVMClauseLists &Lists = getListsForDirective(DK);

  // If this isn't on the list, this is fine.
  if (!Lists.AllowedExclusive.isSet(LLVMClause))
    return false;

  for (const OpenACCClause *C : Clauses) {
    llvm::acc::Clause ExistingLLVMClause =
        getLLVMClauseFromClangClause(C->getClauseKind());
    if (Lists.AllowedExclusive.isSet(ExistingLLVMClause)) {
      Diag(ClauseLoc, diag::err_acc_clause_cannot_combine)
          << CK << C->getClauseKind() << DK;
      Diag(C->getBeginLoc(), diag::note_acc_previous_clause_here);

      return true;
    }
  }

  return false;
}

// Diagnoses if 'CK' is not allowed on a directive of 'DK'.
bool SemaOpenACC::DiagnoseAllowedClauses(OpenACCDirectiveKind DK,
                                         OpenACCClauseKind CK,
                                         SourceLocation ClauseLoc) {
  if (DK == OpenACCDirectiveKind::Invalid || CK == OpenACCClauseKind::Invalid)
    return false;
  const LLVMClauseLists &Lists = getListsForDirective(DK);
  llvm::acc::Clause Clause = getLLVMClauseFromClangClause(CK);

  if (!Lists.Allowed.isSet(Clause) && !Lists.AllowedOnce.isSet(Clause) &&
      !Lists.AllowedExclusive.isSet(Clause) && !Lists.Required.isSet(Clause))
    return Diag(ClauseLoc, diag::err_acc_clause_appertainment) << DK << CK;

  return false;
}
