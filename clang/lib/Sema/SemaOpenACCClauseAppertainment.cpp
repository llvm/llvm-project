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

using namespace clang;

namespace {
// Implements a simple 'enum-set' which stores enum values in a single 64 bit
// value. Flang has `EnumSet` which is pretty sizable/has a lot of dependencies,
// so likely not worth bringing in for this use.
class AccClauseSet {
  // We're just using a uint64_t as our underlying rep, so if this size ever
  // gets bigger than 64, we probably need a pair of uint64_ts.
  static_assert(static_cast<unsigned>(OpenACCClauseKind::Invalid) < 64);
  uint64_t Data;

  void setBit(OpenACCClauseKind C) {
    Data |= static_cast<uint64_t>(1) << static_cast<uint64_t>(C);
  }

public:
  constexpr AccClauseSet(
      const std::initializer_list<OpenACCClauseKind> &Clauses)
      : Data(0) {
    for (OpenACCClauseKind C : Clauses)
      setBit(C);
  }

  constexpr bool isSet(OpenACCClauseKind C) const {
    return ((Data >> static_cast<uint64_t>(C)) & 1) != 0;
  }

  void clearBit(OpenACCClauseKind C) {
    Data &= ~(static_cast<uint64_t>(1) << static_cast<uint64_t>(C));
  }

  constexpr bool isEmpty() const { return Data == 0; }

  unsigned popcount() const { return llvm::popcount<uint64_t>(Data); }
};

struct LLVMClauseLists {
  AccClauseSet Allowed;
  AccClauseSet AllowedOnce;
  AccClauseSet AllowedExclusive;
  AccClauseSet Required;
};
struct LLVMDirectiveClauseRelationships {
  OpenACCDirectiveKind DirKind;
  LLVMClauseLists Lists;
};

} // namespace

// This introduces these in a llvm::acc namespace, so make sure this stays in
// the global namespace.
#define GEN_CLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenACC/ACC.inc"

namespace {
LLVMDirectiveClauseRelationships Relations[] =
#define GEN_CLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenACC/ACC.inc"
    ;

const LLVMClauseLists &getListsForDirective(OpenACCDirectiveKind DK) {

  auto Res = llvm::find_if(Relations,
                           [=](const LLVMDirectiveClauseRelationships &Rel) {
                             return Rel.DirKind == DK;
                           });
  assert(Res != std::end(Relations) && "Unknown directive kind?");

  return Res->Lists;
}

std::string getListOfClauses(AccClauseSet Set) {
  // We could probably come up with a better way to do this smuggling, but this
  // is good enough for now.
  std::string Output;
  llvm::raw_string_ostream OS{Output};

  for (unsigned I = 0; I < static_cast<unsigned>(OpenACCClauseKind::Invalid);
       ++I) {
    OpenACCClauseKind CurClause = static_cast<OpenACCClauseKind>(I);
    if (!Set.isSet(CurClause))
      continue;

    OS << '\'' << CurClause << '\'';

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

OpenACCClauseKind dealiasClauseKind(OpenACCClauseKind CK) {
  switch (CK) {
  default:
    return CK;
#define VISIT_CLAUSE(NAME)
#define CLAUSE_ALIAS(ALIAS, NAME, DEPRECATED)                                  \
  case OpenACCClauseKind::ALIAS:                                               \
    return OpenACCClauseKind::NAME;
#include "clang/Basic/OpenACCClauses.def"
  }

  return CK;
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
    if (Lists.Required.isSet(dealiasClauseKind(C->getClauseKind())))
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

  OpenACCClauseKind Dealiased = dealiasClauseKind(CK);

  const LLVMClauseLists &Lists = getListsForDirective(DK);
  if (!Lists.AllowedOnce.isSet(CK))
    return false;

  auto Res = llvm::find_if(Clauses, [=](const OpenACCClause *C) {
    return dealiasClauseKind(C->getClauseKind()) == Dealiased;
  });

  if (Res == Clauses.end())
    return false;

  Diag(ClauseLoc, diag::err_acc_duplicate_clause_disallowed) << DK << CK;
  Diag((*Res)->getBeginLoc(), diag::note_acc_previous_clause_here) << CK;
  return true;
}

// Diagnoses a 'CK' on a 'DK' being added that isn't allowed to, because another
// clause in 'Clauses' already exists.
bool SemaOpenACC::DiagnoseExclusiveClauses(
    OpenACCDirectiveKind DK, OpenACCClauseKind CK, SourceLocation ClauseLoc,
    ArrayRef<const OpenACCClause *> Clauses) {
  if (DK == OpenACCDirectiveKind::Invalid || CK == OpenACCClauseKind::Invalid)
    return false;

  const LLVMClauseLists &Lists = getListsForDirective(DK);
  OpenACCClauseKind Dealiased = dealiasClauseKind(CK);

  // If this isn't on the list, this is fine.
  if (!Lists.AllowedExclusive.isSet(Dealiased))
    return false;

  for (const OpenACCClause *C : Clauses) {
    if (Lists.AllowedExclusive.isSet(dealiasClauseKind(C->getClauseKind()))) {
      Diag(ClauseLoc, diag::err_acc_clause_cannot_combine)
          << CK << C->getClauseKind() << DK;
      Diag(C->getBeginLoc(), diag::note_acc_previous_clause_here)
          << C->getClauseKind();

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
  OpenACCClauseKind Dealiased = dealiasClauseKind(CK);

  if (!Lists.Allowed.isSet(Dealiased) && !Lists.AllowedOnce.isSet(Dealiased) &&
      !Lists.AllowedExclusive.isSet(Dealiased) &&
      !Lists.Required.isSet(Dealiased))
    return Diag(ClauseLoc, diag::err_acc_clause_appertainment) << DK << CK;

  return false;
}
