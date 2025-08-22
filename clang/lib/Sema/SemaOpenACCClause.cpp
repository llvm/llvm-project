//===--- SemaOpenACCClause.cpp - Semantic Analysis for OpenACC clause -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenACC clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/SemaOpenACC.h"

using namespace clang;

namespace {
bool checkValidAfterDeviceType(
    SemaOpenACC &S, const OpenACCDeviceTypeClause &DeviceTypeClause,
    const SemaOpenACC::OpenACCParsedClause &NewClause) {
  // OpenACC3.3: Section 2.4: Clauses that precede any device_type clause are
  // default clauses.  Clauses that follow a device_type clause up to the end of
  // the directive or up to the next device_type clause are device-specific
  // clauses for the device types specified in the device_type argument.
  //
  // The above implies that despite what the individual text says, these are
  // valid.
  if (NewClause.getClauseKind() == OpenACCClauseKind::DType ||
      NewClause.getClauseKind() == OpenACCClauseKind::DeviceType)
    return false;

  // Implement check from OpenACC3.3: section 2.5.4:
  // Only the async, wait, num_gangs, num_workers, and vector_length clauses may
  // follow a device_type clause.
  if (isOpenACCComputeDirectiveKind(NewClause.getDirectiveKind())) {
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Async:
    case OpenACCClauseKind::Wait:
    case OpenACCClauseKind::NumGangs:
    case OpenACCClauseKind::NumWorkers:
    case OpenACCClauseKind::VectorLength:
      return false;
    default:
      break;
    }
  } else if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Loop) {
    // Implement check from OpenACC3.3: section 2.9:
    // Only the collapse, gang, worker, vector, seq, independent, auto, and tile
    // clauses may follow a device_type clause.
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Collapse:
    case OpenACCClauseKind::Gang:
    case OpenACCClauseKind::Worker:
    case OpenACCClauseKind::Vector:
    case OpenACCClauseKind::Seq:
    case OpenACCClauseKind::Independent:
    case OpenACCClauseKind::Auto:
    case OpenACCClauseKind::Tile:
      return false;
    default:
      break;
    }
  } else if (isOpenACCCombinedDirectiveKind(NewClause.getDirectiveKind())) {
    // This seems like it should be the union of 2.9 and 2.5.4 from above.
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Async:
    case OpenACCClauseKind::Wait:
    case OpenACCClauseKind::NumGangs:
    case OpenACCClauseKind::NumWorkers:
    case OpenACCClauseKind::VectorLength:
    case OpenACCClauseKind::Collapse:
    case OpenACCClauseKind::Gang:
    case OpenACCClauseKind::Worker:
    case OpenACCClauseKind::Vector:
    case OpenACCClauseKind::Seq:
    case OpenACCClauseKind::Independent:
    case OpenACCClauseKind::Auto:
    case OpenACCClauseKind::Tile:
      return false;
    default:
      break;
    }
  } else if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Data) {
    // OpenACC3.3 section 2.6.5: Only the async and wait clauses may follow a
    // device_type clause.
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Async:
    case OpenACCClauseKind::Wait:
      return false;
    default:
      break;
    }
  } else if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Set ||
             NewClause.getDirectiveKind() == OpenACCDirectiveKind::Init ||
             NewClause.getDirectiveKind() == OpenACCDirectiveKind::Shutdown) {
    // There are no restrictions on 'set', 'init', or 'shutdown'.
    return false;
  } else if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Update) {
    // OpenACC3.3 section 2.14.4: Only the async and wait clauses may follow a
    // device_type clause.
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Async:
    case OpenACCClauseKind::Wait:
      return false;
    default:
      break;
    }
  } else if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Routine) {
    // OpenACC 3.3 section 2.15: Only the 'gang', 'worker', 'vector', 'seq', and
    // 'bind' clauses may follow a device_type clause.
    switch (NewClause.getClauseKind()) {
    case OpenACCClauseKind::Gang:
    case OpenACCClauseKind::Worker:
    case OpenACCClauseKind::Vector:
    case OpenACCClauseKind::Seq:
    case OpenACCClauseKind::Bind:
      return false;
    default:
      break;
    }
  }
  S.Diag(NewClause.getBeginLoc(), diag::err_acc_clause_after_device_type)
      << NewClause.getClauseKind() << DeviceTypeClause.getClauseKind()
      << NewClause.getDirectiveKind();
  S.Diag(DeviceTypeClause.getBeginLoc(),
         diag::note_acc_active_applies_clause_here)
      << diag::ACCDeviceTypeApp::Active << DeviceTypeClause.getClauseKind();
  return true;
}

// GCC looks through linkage specs, but not the other transparent declaration
// contexts for 'declare' restrictions, so this helper function helps get us
// through that.
const DeclContext *removeLinkageSpecDC(const DeclContext *DC) {
  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  return DC;
}

class SemaOpenACCClauseVisitor {
  SemaOpenACC &SemaRef;
  ASTContext &Ctx;
  ArrayRef<const OpenACCClause *> ExistingClauses;

  // OpenACC 3.3 2.9:
  //  A 'gang', 'worker', or 'vector' clause may not appear if a 'seq' clause
  //  appears.
  bool
  DiagGangWorkerVectorSeqConflict(SemaOpenACC::OpenACCParsedClause &Clause) {
    if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Loop &&
        !isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind()))
      return false;
    assert(Clause.getClauseKind() == OpenACCClauseKind::Gang ||
           Clause.getClauseKind() == OpenACCClauseKind::Worker ||
           Clause.getClauseKind() == OpenACCClauseKind::Vector);
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCSeqClause>);

    if (Itr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_cannot_combine)
          << Clause.getClauseKind() << (*Itr)->getClauseKind()
          << Clause.getDirectiveKind();
      SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
          << (*Itr)->getClauseKind();

      return true;
    }
    return false;
  }

  OpenACCModifierKind
  CheckModifierList(SemaOpenACC::OpenACCParsedClause &Clause,
                    OpenACCModifierKind Mods) {
    auto CheckSingle = [=](OpenACCModifierKind CurMods,
                           OpenACCModifierKind ValidKinds,
                           OpenACCModifierKind Bit) {
      if (!isOpenACCModifierBitSet(CurMods, Bit) ||
          isOpenACCModifierBitSet(ValidKinds, Bit))
        return CurMods;

      SemaRef.Diag(Clause.getLParenLoc(), diag::err_acc_invalid_modifier)
          << Bit << Clause.getClauseKind();

      return CurMods ^ Bit;
    };
    auto Check = [&](OpenACCModifierKind ValidKinds) {
      if ((Mods | ValidKinds) == ValidKinds)
        return Mods;

      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::Always);
      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::AlwaysIn);
      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::AlwaysOut);
      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::Readonly);
      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::Zero);
      Mods = CheckSingle(Mods, ValidKinds, OpenACCModifierKind::Capture);
      return Mods;
    };

    // The 'capture' modifier is only valid on copyin, copyout, and create on
    // structured data or compute constructs (which also includes combined).
    bool IsStructuredDataOrCompute =
        Clause.getDirectiveKind() == OpenACCDirectiveKind::Data ||
        isOpenACCComputeDirectiveKind(Clause.getDirectiveKind()) ||
        isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind());

    switch (Clause.getClauseKind()) {
    default:
      llvm_unreachable("Only for copy, copyin, copyout, create");
    case OpenACCClauseKind::Copy:
    case OpenACCClauseKind::PCopy:
    case OpenACCClauseKind::PresentOrCopy:
      // COPY: Capture always
      return Check(OpenACCModifierKind::Always | OpenACCModifierKind::AlwaysIn |
                   OpenACCModifierKind::AlwaysOut |
                   OpenACCModifierKind::Capture);
    case OpenACCClauseKind::CopyIn:
    case OpenACCClauseKind::PCopyIn:
    case OpenACCClauseKind::PresentOrCopyIn:
      // COPYIN: Capture only struct.data & compute
      return Check(OpenACCModifierKind::Always | OpenACCModifierKind::AlwaysIn |
                   OpenACCModifierKind::Readonly |
                   (IsStructuredDataOrCompute ? OpenACCModifierKind::Capture
                                              : OpenACCModifierKind::Invalid));
    case OpenACCClauseKind::CopyOut:
    case OpenACCClauseKind::PCopyOut:
    case OpenACCClauseKind::PresentOrCopyOut:
      // COPYOUT: Capture only struct.data & compute
      return Check(OpenACCModifierKind::Always |
                   OpenACCModifierKind::AlwaysOut | OpenACCModifierKind::Zero |
                   (IsStructuredDataOrCompute ? OpenACCModifierKind::Capture
                                              : OpenACCModifierKind::Invalid));
    case OpenACCClauseKind::Create:
    case OpenACCClauseKind::PCreate:
    case OpenACCClauseKind::PresentOrCreate:
      // CREATE: Capture only struct.data & compute
      return Check(OpenACCModifierKind::Zero |
                   (IsStructuredDataOrCompute ? OpenACCModifierKind::Capture
                                              : OpenACCModifierKind::Invalid));
    }
    llvm_unreachable("didn't return from switch above?");
  }

  // Helper for the 'routine' checks during 'new' clause addition. Precondition
  // is that we already know the new clause is one of the prohbiited ones.
  template <typename Pred>
  bool
  CheckValidRoutineNewClauseHelper(Pred HasPredicate,
                                   SemaOpenACC::OpenACCParsedClause &Clause) {
    if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Routine)
      return false;

    auto *FirstDeviceType =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCDeviceTypeClause>);

    if (FirstDeviceType == ExistingClauses.end()) {
      // If there isn't a device type yet, ANY duplicate is wrong.

      auto *ExistingProhibitedClause =
          llvm::find_if(ExistingClauses, HasPredicate);

      if (ExistingProhibitedClause == ExistingClauses.end())
        return false;

      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_cannot_combine)
          << Clause.getClauseKind()
          << (*ExistingProhibitedClause)->getClauseKind()
          << Clause.getDirectiveKind();
      SemaRef.Diag((*ExistingProhibitedClause)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*ExistingProhibitedClause)->getClauseKind();
      return true;
    }

    // At this point we know that this is 'after' a device type. So this is an
    // error if: 1- there is one BEFORE the 'device_type' 2- there is one
    // between this and the previous 'device_type'.

    auto *BeforeDeviceType =
        std::find_if(ExistingClauses.begin(), FirstDeviceType, HasPredicate);
    // If there is one before the device_type (and we know we are after a
    // device_type), than this is ill-formed.
    if (BeforeDeviceType != FirstDeviceType) {
      SemaRef.Diag(
          Clause.getBeginLoc(),
          diag::err_acc_clause_routine_cannot_combine_before_device_type)
          << Clause.getClauseKind() << (*BeforeDeviceType)->getClauseKind();
      SemaRef.Diag((*BeforeDeviceType)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*BeforeDeviceType)->getClauseKind();
      SemaRef.Diag((*FirstDeviceType)->getBeginLoc(),
                   diag::note_acc_active_applies_clause_here)
          << diag::ACCDeviceTypeApp::Active
          << (*FirstDeviceType)->getClauseKind();
      return true;
    }

    auto LastDeviceTypeItr =
        std::find_if(ExistingClauses.rbegin(), ExistingClauses.rend(),
                     llvm::IsaPred<OpenACCDeviceTypeClause>);

    // We already know there is one in the list, so it is nonsensical to not
    // have one.
    assert(LastDeviceTypeItr != ExistingClauses.rend());

    // Get the device-type from-the-front (not reverse) iterator from the
    // reverse iterator.
    auto *LastDeviceType = LastDeviceTypeItr.base() - 1;

    auto *ExistingProhibitedSinceLastDevice =
        std::find_if(LastDeviceType, ExistingClauses.end(), HasPredicate);

    // No prohibited ones since the last device-type.
    if (ExistingProhibitedSinceLastDevice == ExistingClauses.end())
      return false;

    SemaRef.Diag(Clause.getBeginLoc(),
                 diag::err_acc_clause_routine_cannot_combine_same_device_type)
        << Clause.getClauseKind()
        << (*ExistingProhibitedSinceLastDevice)->getClauseKind();
    SemaRef.Diag((*ExistingProhibitedSinceLastDevice)->getBeginLoc(),
                 diag::note_acc_previous_clause_here)
        << (*ExistingProhibitedSinceLastDevice)->getClauseKind();
    SemaRef.Diag((*LastDeviceType)->getBeginLoc(),
                 diag::note_acc_active_applies_clause_here)
        << diag::ACCDeviceTypeApp::Active << (*LastDeviceType)->getClauseKind();
    return true;
  }

  // Routine has a pretty complicated set of rules for how device_type and the
  // gang, worker, vector, and seq clauses work.  So diagnose some of it here.
  bool CheckValidRoutineGangWorkerVectorSeqNewClause(
      SemaOpenACC::OpenACCParsedClause &Clause) {

    if (Clause.getClauseKind() != OpenACCClauseKind::Gang &&
        Clause.getClauseKind() != OpenACCClauseKind::Vector &&
        Clause.getClauseKind() != OpenACCClauseKind::Worker &&
        Clause.getClauseKind() != OpenACCClauseKind::Seq)
      return false;
    auto ProhibitedPred = llvm::IsaPred<OpenACCGangClause, OpenACCWorkerClause,
                                        OpenACCVectorClause, OpenACCSeqClause>;

    return CheckValidRoutineNewClauseHelper(ProhibitedPred, Clause);
  }

  // Bind should have similar rules on a routine as gang/worker/vector/seq,
  // except there is no 'must have 1' rule, so we can get all the checking done
  // here.
  bool
  CheckValidRoutineBindNewClause(SemaOpenACC::OpenACCParsedClause &Clause) {

    if (Clause.getClauseKind() != OpenACCClauseKind::Bind)
      return false;

    auto HasBindPred = llvm::IsaPred<OpenACCBindClause>;
    return CheckValidRoutineNewClauseHelper(HasBindPred, Clause);
  }

  // For 'tile' and 'collapse', only allow 1 per 'device_type'.
  // Also applies to num_worker, num_gangs, vector_length, and async.
  // This does introspection into the actual device-types to prevent duplicates
  // across device types as well.
  template <typename TheClauseTy>
  bool DisallowSinceLastDeviceType(SemaOpenACC::OpenACCParsedClause &Clause) {
    auto LastDeviceTypeItr =
        std::find_if(ExistingClauses.rbegin(), ExistingClauses.rend(),
                     llvm::IsaPred<OpenACCDeviceTypeClause>);

    auto LastSinceDevTy =
        std::find_if(ExistingClauses.rbegin(), LastDeviceTypeItr,
                     llvm::IsaPred<TheClauseTy>);

    // In this case there is a duplicate since the last device_type/lack of a
    // device_type.  Diagnose these as duplicates.
    if (LastSinceDevTy != LastDeviceTypeItr) {
      SemaRef.Diag(Clause.getBeginLoc(),
                   diag::err_acc_clause_since_last_device_type)
          << Clause.getClauseKind() << Clause.getDirectiveKind()
          << (LastDeviceTypeItr != ExistingClauses.rend());

      SemaRef.Diag((*LastSinceDevTy)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*LastSinceDevTy)->getClauseKind();

      // Mention the last device_type as well.
      if (LastDeviceTypeItr != ExistingClauses.rend())
        SemaRef.Diag((*LastDeviceTypeItr)->getBeginLoc(),
                     diag::note_acc_active_applies_clause_here)
            << diag::ACCDeviceTypeApp::Active
            << (*LastDeviceTypeItr)->getClauseKind();
      return true;
    }

    // If this isn't in a device_type, and we didn't diagnose that there are
    // dupes above, just give up, no sense in searching for previous device_type
    // regions as they don't exist.
    if (LastDeviceTypeItr == ExistingClauses.rend())
      return false;

    // The device-type that is active for us, so we can compare to the previous
    // ones.
    const auto &ActiveDeviceTypeClause =
        cast<OpenACCDeviceTypeClause>(**LastDeviceTypeItr);

    auto PrevDeviceTypeItr = LastDeviceTypeItr;
    auto CurDevTypeItr = LastDeviceTypeItr;

    while ((CurDevTypeItr = std::find_if(
                std::next(PrevDeviceTypeItr), ExistingClauses.rend(),
                llvm::IsaPred<OpenACCDeviceTypeClause>)) !=
           ExistingClauses.rend()) {
      // At this point, we know that we have a region between two device_types,
      // as specified by CurDevTypeItr and PrevDeviceTypeItr.

      auto CurClauseKindItr = std::find_if(PrevDeviceTypeItr, CurDevTypeItr,
                                           llvm::IsaPred<TheClauseTy>);

      // There are no clauses of the current kind between these device_types, so
      // continue.
      if (CurClauseKindItr == CurDevTypeItr) {
        PrevDeviceTypeItr = CurDevTypeItr;
        continue;
      }

      // At this point, we know that this device_type region has a collapse.  So
      // diagnose if the two device_types have any overlap in their
      // architectures.
      const auto &CurDeviceTypeClause =
          cast<OpenACCDeviceTypeClause>(**CurDevTypeItr);

      for (const DeviceTypeArgument &arg :
           ActiveDeviceTypeClause.getArchitectures()) {
        for (const DeviceTypeArgument &prevArg :
             CurDeviceTypeClause.getArchitectures()) {

          // This should catch duplicates * regions, duplicate same-text (thanks
          // to identifier equiv.) and case insensitive dupes.
          if (arg.getIdentifierInfo() == prevArg.getIdentifierInfo() ||
              (arg.getIdentifierInfo() && prevArg.getIdentifierInfo() &&
               StringRef{arg.getIdentifierInfo()->getName()}.equals_insensitive(
                   prevArg.getIdentifierInfo()->getName()))) {
            SemaRef.Diag(Clause.getBeginLoc(),
                         diag::err_acc_clause_conflicts_prev_dev_type)
                << Clause.getClauseKind()
                << (arg.getIdentifierInfo() ? arg.getIdentifierInfo()->getName()
                                            : "*");
            // mention the active device type.
            SemaRef.Diag(ActiveDeviceTypeClause.getBeginLoc(),
                         diag::note_acc_active_applies_clause_here)
                << diag::ACCDeviceTypeApp::Active
                << ActiveDeviceTypeClause.getClauseKind();
            // mention the previous clause.
            SemaRef.Diag((*CurClauseKindItr)->getBeginLoc(),
                         diag::note_acc_previous_clause_here)
                << (*CurClauseKindItr)->getClauseKind();
            // mention the previous device type.
            SemaRef.Diag(CurDeviceTypeClause.getBeginLoc(),
                         diag::note_acc_active_applies_clause_here)
                << diag::ACCDeviceTypeApp::Applies
                << CurDeviceTypeClause.getClauseKind();
            return true;
          }
        }
      }

      PrevDeviceTypeItr = CurDevTypeItr;
    }
    return false;
  }

public:
  SemaOpenACCClauseVisitor(SemaOpenACC &S,
                           ArrayRef<const OpenACCClause *> ExistingClauses)
      : SemaRef(S), Ctx(S.getASTContext()), ExistingClauses(ExistingClauses) {}

  OpenACCClause *Visit(SemaOpenACC::OpenACCParsedClause &Clause) {

    if (SemaRef.DiagnoseAllowedOnceClauses(
            Clause.getDirectiveKind(), Clause.getClauseKind(),
            Clause.getBeginLoc(), ExistingClauses) ||
        SemaRef.DiagnoseExclusiveClauses(Clause.getDirectiveKind(),
                                         Clause.getClauseKind(),
                                         Clause.getBeginLoc(), ExistingClauses))
      return nullptr;
    if (CheckValidRoutineGangWorkerVectorSeqNewClause(Clause) ||
        CheckValidRoutineBindNewClause(Clause))
      return nullptr;

    switch (Clause.getClauseKind()) {
    case OpenACCClauseKind::Shortloop:
      llvm_unreachable("Shortloop shouldn't be generated in clang");
    case OpenACCClauseKind::Invalid:
      return nullptr;
#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  case OpenACCClauseKind::CLAUSE_NAME:                                         \
    return Visit##CLAUSE_NAME##Clause(Clause);
#define CLAUSE_ALIAS(ALIAS, CLAUSE_NAME, DEPRECATED)                           \
  case OpenACCClauseKind::ALIAS:                                               \
  if (DEPRECATED)                                                              \
    SemaRef.Diag(Clause.getBeginLoc(), diag::warn_acc_deprecated_alias_name)   \
        << Clause.getClauseKind() << OpenACCClauseKind::CLAUSE_NAME;           \
  return Visit##CLAUSE_NAME##Clause(Clause);
#include "clang/Basic/OpenACCClauses.def"
    }
    llvm_unreachable("Invalid clause kind");
  }

#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  OpenACCClause *Visit##CLAUSE_NAME##Clause(                                   \
      SemaOpenACC::OpenACCParsedClause &Clause);
#include "clang/Basic/OpenACCClauses.def"
};

OpenACCClause *SemaOpenACCClauseVisitor::VisitDefaultClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // Don't add an invalid clause to the AST.
  if (Clause.getDefaultClauseKind() == OpenACCDefaultClauseKind::Invalid)
    return nullptr;

  return OpenACCDefaultClause::Create(
      Ctx, Clause.getDefaultClauseKind(), Clause.getBeginLoc(),
      Clause.getLParenLoc(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitTileClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DisallowSinceLastDeviceType<OpenACCTileClause>(Clause))
    return nullptr;

  llvm::SmallVector<Expr *> NewSizeExprs;

  // Make sure these are all positive constant expressions or *.
  for (Expr *E : Clause.getIntExprs()) {
    ExprResult Res = SemaRef.CheckTileSizeExpr(E);

    if (!Res.isUsable())
      return nullptr;

    NewSizeExprs.push_back(Res.get());
  }

  return OpenACCTileClause::Create(Ctx, Clause.getBeginLoc(),
                                   Clause.getLParenLoc(), NewSizeExprs,
                                   Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitIfClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  // The parser has ensured that we have a proper condition expr, so there
  // isn't really much to do here.

  // If the 'if' clause is true, it makes the 'self' clause have no effect,
  // diagnose that here.  This only applies on compute/combined constructs.
  if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Update) {
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCSelfClause>);
    if (Itr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(), diag::warn_acc_if_self_conflict);
      SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
          << (*Itr)->getClauseKind();
    }
  }

  return OpenACCIfClause::Create(Ctx, Clause.getBeginLoc(),
                                 Clause.getLParenLoc(),
                                 Clause.getConditionExpr(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitSelfClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  // If the 'if' clause is true, it makes the 'self' clause have no effect,
  // diagnose that here.  This only applies on compute/combined constructs.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Update)
    return OpenACCSelfClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), Clause.getVarList(),
                                     Clause.getEndLoc());

  const auto *Itr =
      llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCIfClause>);
  if (Itr != ExistingClauses.end()) {
    SemaRef.Diag(Clause.getBeginLoc(), diag::warn_acc_if_self_conflict);
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
        << (*Itr)->getClauseKind();
  }
  return OpenACCSelfClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getConditionExpr(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitNumGangsClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DisallowSinceLastDeviceType<OpenACCNumGangsClause>(Clause))
    return nullptr;

  // num_gangs requires at least 1 int expr in all forms.  Diagnose here, but
  // allow us to continue, an empty clause might be useful for future
  // diagnostics.
  if (Clause.getIntExprs().empty())
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_num_gangs_num_args)
        << /*NoArgs=*/0;

  unsigned MaxArgs =
      (Clause.getDirectiveKind() == OpenACCDirectiveKind::Parallel ||
       Clause.getDirectiveKind() == OpenACCDirectiveKind::ParallelLoop)
          ? 3
          : 1;
  // The max number of args differs between parallel and other constructs.
  // Again, allow us to continue for the purposes of future diagnostics.
  if (Clause.getIntExprs().size() > MaxArgs)
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_num_gangs_num_args)
        << /*NoArgs=*/1 << Clause.getDirectiveKind() << MaxArgs
        << Clause.getIntExprs().size();

  // OpenACC 3.3 Section 2.9.11: A reduction clause may not appear on a loop
  // directive that has a gang clause and is within a compute construct that has
  // a num_gangs clause with more than one explicit argument.
  if (Clause.getIntExprs().size() > 1 &&
      isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
    auto *GangClauseItr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCGangClause>);
    auto *ReductionClauseItr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCReductionClause>);

    if (GangClauseItr != ExistingClauses.end() &&
        ReductionClauseItr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(),
                   diag::err_acc_gang_reduction_numgangs_conflict)
          << OpenACCClauseKind::Reduction << OpenACCClauseKind::Gang
          << Clause.getDirectiveKind() << /*is on combined directive=*/1;
      SemaRef.Diag((*ReductionClauseItr)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*ReductionClauseItr)->getClauseKind();
      SemaRef.Diag((*GangClauseItr)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*GangClauseItr)->getClauseKind();
      return nullptr;
    }
  }

  // OpenACC 3.3 Section 2.5.4:
  // A reduction clause may not appear on a parallel construct with a
  // num_gangs clause that has more than one argument.
  if ((Clause.getDirectiveKind() == OpenACCDirectiveKind::Parallel ||
       Clause.getDirectiveKind() == OpenACCDirectiveKind::ParallelLoop) &&
      Clause.getIntExprs().size() > 1) {
    auto *Parallel =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCReductionClause>);

    if (Parallel != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(),
                   diag::err_acc_reduction_num_gangs_conflict)
          << /*>1 arg in first loc=*/1 << Clause.getClauseKind()
          << Clause.getDirectiveKind() << OpenACCClauseKind::Reduction;
      SemaRef.Diag((*Parallel)->getBeginLoc(),
                   diag::note_acc_previous_clause_here)
          << (*Parallel)->getClauseKind();
      return nullptr;
    }
  }

  // OpenACC 3.3 Section 2.9.2:
  // An argument with no keyword or with the 'num' keyword is allowed only when
  // the 'num_gangs' does not appear on the 'kernel' construct.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::KernelsLoop) {
    auto GangClauses = llvm::make_filter_range(
        ExistingClauses, llvm::IsaPred<OpenACCGangClause>);

    for (auto *GC : GangClauses) {
      if (cast<OpenACCGangClause>(GC)->hasExprOfKind(OpenACCGangKind::Num)) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_num_arg_conflict_reverse)
            << OpenACCClauseKind::NumGangs << OpenACCClauseKind::Gang
            << /*Num argument*/ 1;
        SemaRef.Diag(GC->getBeginLoc(), diag::note_acc_previous_clause_here)
            << GC->getClauseKind();
        return nullptr;
      }
    }
  }

  return OpenACCNumGangsClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs(),
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitNumWorkersClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DisallowSinceLastDeviceType<OpenACCNumWorkersClause>(Clause))
    return nullptr;

  // OpenACC 3.3 Section 2.9.2:
  // An argument is allowed only when the 'num_workers' does not appear on the
  // kernels construct.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::KernelsLoop) {
    auto WorkerClauses = llvm::make_filter_range(
        ExistingClauses, llvm::IsaPred<OpenACCWorkerClause>);

    for (auto *WC : WorkerClauses) {
      if (cast<OpenACCWorkerClause>(WC)->hasIntExpr()) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_num_arg_conflict_reverse)
            << OpenACCClauseKind::NumWorkers << OpenACCClauseKind::Worker
            << /*num argument*/ 0;
        SemaRef.Diag(WC->getBeginLoc(), diag::note_acc_previous_clause_here)
            << WC->getClauseKind();
        return nullptr;
      }
    }
  }

  assert(Clause.getIntExprs().size() == 1 &&
         "Invalid number of expressions for NumWorkers");
  return OpenACCNumWorkersClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs()[0],
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitVectorLengthClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DisallowSinceLastDeviceType<OpenACCVectorLengthClause>(Clause))
    return nullptr;

  // OpenACC 3.3 Section 2.9.4:
  // An argument is allowed only when the 'vector_length' does not appear on the
  // 'kernels' construct.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::KernelsLoop) {
    auto VectorClauses = llvm::make_filter_range(
        ExistingClauses, llvm::IsaPred<OpenACCVectorClause>);

    for (auto *VC : VectorClauses) {
      if (cast<OpenACCVectorClause>(VC)->hasIntExpr()) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_num_arg_conflict_reverse)
            << OpenACCClauseKind::VectorLength << OpenACCClauseKind::Vector
            << /*num argument*/ 0;
        SemaRef.Diag(VC->getBeginLoc(), diag::note_acc_previous_clause_here)
            << VC->getClauseKind();
        return nullptr;
      }
    }
  }

  assert(Clause.getIntExprs().size() == 1 &&
         "Invalid number of expressions for NumWorkers");
  return OpenACCVectorLengthClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs()[0],
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitAsyncClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DisallowSinceLastDeviceType<OpenACCAsyncClause>(Clause))
    return nullptr;

  assert(Clause.getNumIntExprs() < 2 &&
         "Invalid number of expressions for Async");
  return OpenACCAsyncClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getNumIntExprs() != 0 ? Clause.getIntExprs()[0] : nullptr,
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDeviceNumClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  assert(Clause.getNumIntExprs() == 1 &&
         "Invalid number of expressions for device_num");
  return OpenACCDeviceNumClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs()[0],
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDefaultAsyncClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  assert(Clause.getNumIntExprs() == 1 &&
         "Invalid number of expressions for default_async");
  return OpenACCDefaultAsyncClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs()[0],
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitPrivateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  llvm::SmallVector<VarDecl *> InitRecipes;

  // Assemble the recipes list.
  for (const Expr *VarExpr : Clause.getVarList())
    InitRecipes.push_back(
        SemaRef.CreateInitRecipe(OpenACCClauseKind::Private, VarExpr).first);

  return OpenACCPrivateClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      InitRecipes, Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitFirstPrivateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  llvm::SmallVector<OpenACCFirstPrivateRecipe> InitRecipes;

  // Assemble the recipes list.
  for (const Expr *VarExpr : Clause.getVarList())
    InitRecipes.push_back(
        SemaRef.CreateInitRecipe(OpenACCClauseKind::FirstPrivate, VarExpr));

  return OpenACCFirstPrivateClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      InitRecipes, Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitNoCreateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCNoCreateClause::Create(Ctx, Clause.getBeginLoc(),
                                       Clause.getLParenLoc(),
                                       Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitPresentClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, OpenACCModifierKind::Invalid))
    return nullptr;

  return OpenACCPresentClause::Create(Ctx, Clause.getBeginLoc(),
                                      Clause.getLParenLoc(),
                                      Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitHostClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCHostClause::Create(Ctx, Clause.getBeginLoc(),
                                   Clause.getLParenLoc(), Clause.getVarList(),
                                   Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDeviceClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCDeviceClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), Clause.getVarList(),
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCopyClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  OpenACCModifierKind NewMods =
      CheckModifierList(Clause, Clause.getModifierList());

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, NewMods))
    return nullptr;

  return OpenACCCopyClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getModifierList(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitLinkClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, OpenACCModifierKind::Invalid))
    return nullptr;

  Clause.setVarListDetails(SemaRef.CheckLinkClauseVarList(Clause.getVarList()),
                           OpenACCModifierKind::Invalid);

  return OpenACCLinkClause::Create(Ctx, Clause.getBeginLoc(),
                                   Clause.getLParenLoc(), Clause.getVarList(),
                                   Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDeviceResidentClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, OpenACCModifierKind::Invalid))
    return nullptr;

  return OpenACCDeviceResidentClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCopyInClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  OpenACCModifierKind NewMods =
      CheckModifierList(Clause, Clause.getModifierList());

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, NewMods))
    return nullptr;

  return OpenACCCopyInClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getModifierList(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCopyOutClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  OpenACCModifierKind NewMods =
      CheckModifierList(Clause, Clause.getModifierList());

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, NewMods))
    return nullptr;

  return OpenACCCopyOutClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getModifierList(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCreateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  OpenACCModifierKind NewMods =
      CheckModifierList(Clause, Clause.getModifierList());

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, NewMods))
    return nullptr;

  return OpenACCCreateClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getModifierList(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitAttachClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, but we
  // still have to make sure it is a pointer type.
  llvm::SmallVector<Expr *> VarList{Clause.getVarList()};
  llvm::erase_if(VarList, [&](Expr *E) {
    return SemaRef.CheckVarIsPointerType(OpenACCClauseKind::Attach, E);
  });
  Clause.setVarListDetails(VarList, OpenACCModifierKind::Invalid);
  return OpenACCAttachClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), Clause.getVarList(),
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDetachClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, but we
  // still have to make sure it is a pointer type.
  llvm::SmallVector<Expr *> VarList{Clause.getVarList()};
  llvm::erase_if(VarList, [&](Expr *E) {
    return SemaRef.CheckVarIsPointerType(OpenACCClauseKind::Detach, E);
  });
  Clause.setVarListDetails(VarList, OpenACCModifierKind::Invalid);
  return OpenACCDetachClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), Clause.getVarList(),
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDeleteClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.
  return OpenACCDeleteClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), Clause.getVarList(),
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitUseDeviceClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable or array, so nothing
  // left to do here.
  return OpenACCUseDeviceClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDevicePtrClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, but we
  // still have to make sure it is a pointer type.
  llvm::SmallVector<Expr *> VarList{Clause.getVarList()};
  llvm::erase_if(VarList, [&](Expr *E) {
    return SemaRef.CheckVarIsPointerType(OpenACCClauseKind::DevicePtr, E);
  });
  Clause.setVarListDetails(VarList, OpenACCModifierKind::Invalid);

  // 'declare' has some restrictions that need to be enforced separately, so
  // check it here.
  if (SemaRef.CheckDeclareClause(Clause, OpenACCModifierKind::Invalid))
    return nullptr;

  return OpenACCDevicePtrClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitWaitClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  return OpenACCWaitClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getDevNumExpr(),
      Clause.getQueuesLoc(), Clause.getQueueIdExprs(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDeviceTypeClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  // OpenACC Pull #550 (https://github.com/OpenACC/openacc-spec/pull/550)
  // clarified that Init, Shutdown, and Set only support a single architecture.
  // Though the dialect only requires it for 'set' as far as we know, we'll just
  // implement all 3 here.
  if ((Clause.getDirectiveKind() == OpenACCDirectiveKind::Init ||
       Clause.getDirectiveKind() == OpenACCDirectiveKind::Shutdown ||
       Clause.getDirectiveKind() == OpenACCDirectiveKind::Set) &&
      Clause.getDeviceTypeArchitectures().size() > 1) {
    SemaRef.Diag(Clause.getDeviceTypeArchitectures()[1].getLoc(),
                 diag::err_acc_device_type_multiple_archs)
        << Clause.getDirectiveKind();
    return nullptr;
  }

  // The list of valid device_type values. Flang also has these hardcoded in
  // openacc_parsers.cpp, as there does not seem to be a reliable backend
  // source. The list below is sourced from Flang, though NVC++ supports only
  // 'nvidia', 'host', 'multicore', and 'default'.
  const std::array<llvm::StringLiteral, 6> ValidValues{
      "default", "nvidia", "acc_device_nvidia", "radeon", "host", "multicore"};
  // As an optimization, we have a manually maintained list of valid values
  // below, rather than trying to calculate from above. These should be kept in
  // sync if/when the above list ever changes.
  std::string ValidValuesString =
      "'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'";

  llvm::SmallVector<DeviceTypeArgument> Architectures{
      Clause.getDeviceTypeArchitectures()};

  // The parser has ensured that we either have a single entry of just '*'
  // (represented by a nullptr IdentifierInfo), or a list.

  bool Diagnosed = false;
  auto FilterPred = [&](const DeviceTypeArgument &Arch) {
    // The '*' case.
    if (!Arch.getIdentifierInfo())
      return false;
    return llvm::find_if(ValidValues, [&](StringRef RHS) {
             return Arch.getIdentifierInfo()->getName().equals_insensitive(RHS);
           }) == ValidValues.end();
  };

  auto Diagnose = [&](const DeviceTypeArgument &Arch) {
    Diagnosed = SemaRef.Diag(Arch.getLoc(), diag::err_acc_invalid_default_type)
                << Arch.getIdentifierInfo() << Clause.getClauseKind()
                << ValidValuesString;
  };

  // There aren't stable enumertor versions of 'for-each-then-erase', so do it
  // here.  We DO keep track of whether we diagnosed something to make sure we
  // don't do the 'erase_if' in the event that the first list didn't find
  // anything.
  llvm::for_each(llvm::make_filter_range(Architectures, FilterPred), Diagnose);
  if (Diagnosed)
    llvm::erase_if(Architectures, FilterPred);

  return OpenACCDeviceTypeClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Architectures, Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitAutoClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  return OpenACCAutoClause::Create(Ctx, Clause.getBeginLoc(),
                                   Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitNoHostClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  return OpenACCNoHostClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitIndependentClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  return OpenACCIndependentClause::Create(Ctx, Clause.getBeginLoc(),
                                          Clause.getEndLoc());
}

ExprResult CheckGangStaticExpr(SemaOpenACC &S, Expr *E) {
  if (isa<OpenACCAsteriskSizeExpr>(E))
    return E;
  return S.ActOnIntExpr(OpenACCDirectiveKind::Invalid, OpenACCClauseKind::Gang,
                        E->getBeginLoc(), E);
}

bool IsOrphanLoop(OpenACCDirectiveKind DK, OpenACCDirectiveKind AssocKind) {
  return DK == OpenACCDirectiveKind::Loop &&
         AssocKind == OpenACCDirectiveKind::Invalid;
}

bool HasAssocKind(OpenACCDirectiveKind DK, OpenACCDirectiveKind AssocKind) {
  return DK == OpenACCDirectiveKind::Loop &&
         AssocKind != OpenACCDirectiveKind::Invalid;
}

ExprResult DiagIntArgInvalid(SemaOpenACC &S, Expr *E, OpenACCGangKind GK,
                             OpenACCClauseKind CK, OpenACCDirectiveKind DK,
                             OpenACCDirectiveKind AssocKind) {
  S.Diag(E->getBeginLoc(), diag::err_acc_int_arg_invalid)
      << GK << CK << IsOrphanLoop(DK, AssocKind) << DK
      << HasAssocKind(DK, AssocKind) << AssocKind;
  return ExprError();
}
ExprResult DiagIntArgInvalid(SemaOpenACC &S, Expr *E, StringRef TagKind,
                             OpenACCClauseKind CK, OpenACCDirectiveKind DK,
                             OpenACCDirectiveKind AssocKind) {
  S.Diag(E->getBeginLoc(), diag::err_acc_int_arg_invalid)
      << TagKind << CK << IsOrphanLoop(DK, AssocKind) << DK
      << HasAssocKind(DK, AssocKind) << AssocKind;
  return ExprError();
}

ExprResult CheckGangDimExpr(SemaOpenACC &S, Expr *E) {
  // OpenACC 3.3 2.9.2: When the parent compute construct is a parallel
  // construct, or an orphaned loop construct, the gang clause behaves as
  // follows. ... The dim argument must be a constant positive integer value
  // 1, 2, or 3.
  // -also-
  // OpenACC 3.3 2.15: The 'dim' argument must be a constant positive integer
  // with value 1, 2, or 3.
  if (!E)
    return ExprError();
  ExprResult Res = S.ActOnIntExpr(OpenACCDirectiveKind::Invalid,
                                  OpenACCClauseKind::Gang, E->getBeginLoc(), E);

  if (!Res.isUsable())
    return Res;

  if (Res.get()->isInstantiationDependent())
    return Res;

  std::optional<llvm::APSInt> ICE =
      Res.get()->getIntegerConstantExpr(S.getASTContext());

  if (!ICE || *ICE <= 0 || ICE > 3) {
    S.Diag(Res.get()->getBeginLoc(), diag::err_acc_gang_dim_value)
        << ICE.has_value() << ICE.value_or(llvm::APSInt{}).getExtValue();
    return ExprError();
  }

  return ExprResult{
      ConstantExpr::Create(S.getASTContext(), Res.get(), APValue{*ICE})};
}

ExprResult CheckGangParallelExpr(SemaOpenACC &S, OpenACCDirectiveKind DK,
                                 OpenACCDirectiveKind AssocKind,
                                 OpenACCGangKind GK, Expr *E) {
  switch (GK) {
  case OpenACCGangKind::Static:
    return CheckGangStaticExpr(S, E);
  case OpenACCGangKind::Num:
    // OpenACC 3.3 2.9.2: When the parent compute construct is a parallel
    // construct, or an orphaned loop construct, the gang clause behaves as
    // follows. ... The num argument is not allowed.
    return DiagIntArgInvalid(S, E, GK, OpenACCClauseKind::Gang, DK, AssocKind);
  case OpenACCGangKind::Dim:
    return CheckGangDimExpr(S, E);
  }
  llvm_unreachable("Unknown gang kind in gang parallel check");
}

ExprResult CheckGangKernelsExpr(SemaOpenACC &S,
                                ArrayRef<const OpenACCClause *> ExistingClauses,
                                OpenACCDirectiveKind DK,
                                OpenACCDirectiveKind AssocKind,
                                OpenACCGangKind GK, Expr *E) {
  switch (GK) {
  // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
  // construct, the gang clause behaves as follows. ... The dim argument is
  // not allowed.
  case OpenACCGangKind::Dim:
    return DiagIntArgInvalid(S, E, GK, OpenACCClauseKind::Gang, DK, AssocKind);
  case OpenACCGangKind::Num: {
    // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
    // construct, the gang clause behaves as follows. ... An argument with no
    // keyword or with num keyword is only allowed when num_gangs does not
    // appear on the kernels construct. ... The region of a loop with the gang
    // clause may not contain another loop with a gang clause unless within a
    // nested compute region.

    // If this is a 'combined' construct, search the list of existing clauses.
    // Else we need to search the containing 'kernel'.
    auto Collection = isOpenACCCombinedDirectiveKind(DK)
                          ? ExistingClauses
                          : S.getActiveComputeConstructInfo().Clauses;

    const auto *Itr =
        llvm::find_if(Collection, llvm::IsaPred<OpenACCNumGangsClause>);

    if (Itr != Collection.end()) {
      S.Diag(E->getBeginLoc(), diag::err_acc_num_arg_conflict)
          << "num" << OpenACCClauseKind::Gang << DK
          << HasAssocKind(DK, AssocKind) << AssocKind
          << OpenACCClauseKind::NumGangs;

      S.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
          << (*Itr)->getClauseKind();
      return ExprError();
    }
    return ExprResult{E};
  }
  case OpenACCGangKind::Static:
    return CheckGangStaticExpr(S, E);
  }
  llvm_unreachable("Unknown gang kind in gang kernels check");
}

ExprResult CheckGangSerialExpr(SemaOpenACC &S, OpenACCDirectiveKind DK,
                               OpenACCDirectiveKind AssocKind,
                               OpenACCGangKind GK, Expr *E) {
  switch (GK) {
  // 'dim' and 'num' don't really make sense on serial, and GCC rejects them
  // too, so we disallow them too.
  case OpenACCGangKind::Dim:
  case OpenACCGangKind::Num:
    return DiagIntArgInvalid(S, E, GK, OpenACCClauseKind::Gang, DK, AssocKind);
  case OpenACCGangKind::Static:
    return CheckGangStaticExpr(S, E);
  }
  llvm_unreachable("Unknown gang kind in gang serial check");
}

ExprResult CheckGangRoutineExpr(SemaOpenACC &S, OpenACCDirectiveKind DK,
                                OpenACCDirectiveKind AssocKind,
                                OpenACCGangKind GK, Expr *E) {
  switch (GK) {
    // Only 'dim' is allowed on a routine, so diallow num and static.
  case OpenACCGangKind::Num:
  case OpenACCGangKind::Static:
    return DiagIntArgInvalid(S, E, GK, OpenACCClauseKind::Gang, DK, AssocKind);
  case OpenACCGangKind::Dim:
    return CheckGangDimExpr(S, E);
  }
  llvm_unreachable("Unknown gang kind in gang serial check");
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitVectorClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DiagGangWorkerVectorSeqConflict(Clause))
    return nullptr;

  Expr *IntExpr =
      Clause.getNumIntExprs() != 0 ? Clause.getIntExprs()[0] : nullptr;
  if (IntExpr) {
    switch (Clause.getDirectiveKind()) {
    default:
      llvm_unreachable("Invalid directive kind for this clause");
    case OpenACCDirectiveKind::Loop:
      switch (SemaRef.getActiveComputeConstructInfo().Kind) {
      case OpenACCDirectiveKind::Invalid:
      case OpenACCDirectiveKind::Parallel:
      case OpenACCDirectiveKind::ParallelLoop:
        // No restriction on when 'parallel' can contain an argument.
        break;
      case OpenACCDirectiveKind::Serial:
      case OpenACCDirectiveKind::SerialLoop:
        // GCC disallows this, and there is no real good reason for us to permit
        // it, so disallow until we come up with a use case that makes sense.
        DiagIntArgInvalid(SemaRef, IntExpr, "length", OpenACCClauseKind::Vector,
                          Clause.getDirectiveKind(),
                          SemaRef.getActiveComputeConstructInfo().Kind);
        IntExpr = nullptr;
        break;
      case OpenACCDirectiveKind::Kernels:
      case OpenACCDirectiveKind::KernelsLoop: {
        const auto *Itr =
            llvm::find_if(SemaRef.getActiveComputeConstructInfo().Clauses,
                          llvm::IsaPred<OpenACCVectorLengthClause>);
        if (Itr != SemaRef.getActiveComputeConstructInfo().Clauses.end()) {
          SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
              << "length" << OpenACCClauseKind::Vector
              << Clause.getDirectiveKind()
              << HasAssocKind(Clause.getDirectiveKind(),
                              SemaRef.getActiveComputeConstructInfo().Kind)
              << SemaRef.getActiveComputeConstructInfo().Kind
              << OpenACCClauseKind::VectorLength;
          SemaRef.Diag((*Itr)->getBeginLoc(),
                       diag::note_acc_previous_clause_here)
              << (*Itr)->getClauseKind();

          IntExpr = nullptr;
        }
        break;
      }
      default:
        llvm_unreachable("Non compute construct in active compute construct");
      }
      break;
    case OpenACCDirectiveKind::KernelsLoop: {
      const auto *Itr = llvm::find_if(ExistingClauses,
                                      llvm::IsaPred<OpenACCVectorLengthClause>);
      if (Itr != ExistingClauses.end()) {
        SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
            << "length" << OpenACCClauseKind::Vector
            << Clause.getDirectiveKind()
            << HasAssocKind(Clause.getDirectiveKind(),
                            SemaRef.getActiveComputeConstructInfo().Kind)
            << SemaRef.getActiveComputeConstructInfo().Kind
            << OpenACCClauseKind::VectorLength;
        SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
            << (*Itr)->getClauseKind();

        IntExpr = nullptr;
      }
      break;
    }
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::Routine:
      DiagIntArgInvalid(SemaRef, IntExpr, "length", OpenACCClauseKind::Vector,
                        Clause.getDirectiveKind(),
                        SemaRef.getActiveComputeConstructInfo().Kind);
      IntExpr = nullptr;
      break;
    case OpenACCDirectiveKind::ParallelLoop:
      break;
    case OpenACCDirectiveKind::Invalid:
      // This can happen when the directive was not recognized, but we continued
      // anyway.  Since there is a lot of stuff that can happen (including
      // 'allow anything' in the parallel loop case), just skip all checking and
      // continue.
      break;
    }
  }

  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop) {
    // OpenACC 3.3 2.9.4: The region of a loop with a 'vector' clause may not
    // contain a loop with a gang, worker, or vector clause unless within a
    // nested compute region.
    if (SemaRef.LoopVectorClauseLoc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Vector << OpenACCClauseKind::Vector
          << /*skip kernels construct info*/ 0;
      SemaRef.Diag(SemaRef.LoopVectorClauseLoc,
                   diag::note_acc_previous_clause_here)
          << "vector";
      return nullptr;
    }
  }

  return OpenACCVectorClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), IntExpr,
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitWorkerClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DiagGangWorkerVectorSeqConflict(Clause))
    return nullptr;

  Expr *IntExpr =
      Clause.getNumIntExprs() != 0 ? Clause.getIntExprs()[0] : nullptr;

  if (IntExpr) {
    switch (Clause.getDirectiveKind()) {
    default:
      llvm_unreachable("Invalid directive kind for this clause");
    case OpenACCDirectiveKind::Invalid:
      // This can happen in cases where the directive was not recognized but we
      // continued anyway.  Kernels allows kind of any integer argument, so we
      // can assume it is that (rather than marking the argument invalid like
      // with parallel/serial/routine), and just continue as if nothing
      // happened.  We'll skip the 'kernels' checking vs num-workers, since this
      // MIGHT be something else.
      break;
    case OpenACCDirectiveKind::Loop:
      switch (SemaRef.getActiveComputeConstructInfo().Kind) {
      case OpenACCDirectiveKind::Invalid:
      case OpenACCDirectiveKind::ParallelLoop:
      case OpenACCDirectiveKind::SerialLoop:
      case OpenACCDirectiveKind::Parallel:
      case OpenACCDirectiveKind::Serial:
        DiagIntArgInvalid(SemaRef, IntExpr, OpenACCGangKind::Num,
                          OpenACCClauseKind::Worker, Clause.getDirectiveKind(),
                          SemaRef.getActiveComputeConstructInfo().Kind);
        IntExpr = nullptr;
        break;
      case OpenACCDirectiveKind::KernelsLoop:
      case OpenACCDirectiveKind::Kernels: {
        const auto *Itr =
            llvm::find_if(SemaRef.getActiveComputeConstructInfo().Clauses,
                          llvm::IsaPred<OpenACCNumWorkersClause>);
        if (Itr != SemaRef.getActiveComputeConstructInfo().Clauses.end()) {
          SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
              << "num" << OpenACCClauseKind::Worker << Clause.getDirectiveKind()
              << HasAssocKind(Clause.getDirectiveKind(),
                              SemaRef.getActiveComputeConstructInfo().Kind)
              << SemaRef.getActiveComputeConstructInfo().Kind
              << OpenACCClauseKind::NumWorkers;
          SemaRef.Diag((*Itr)->getBeginLoc(),
                       diag::note_acc_previous_clause_here)
              << (*Itr)->getClauseKind();

          IntExpr = nullptr;
        }
        break;
      }
      default:
        llvm_unreachable("Non compute construct in active compute construct");
      }
      break;
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::Routine:
      DiagIntArgInvalid(SemaRef, IntExpr, OpenACCGangKind::Num,
                        OpenACCClauseKind::Worker, Clause.getDirectiveKind(),
                        SemaRef.getActiveComputeConstructInfo().Kind);
      IntExpr = nullptr;
      break;
    case OpenACCDirectiveKind::KernelsLoop: {
      const auto *Itr = llvm::find_if(ExistingClauses,
                                      llvm::IsaPred<OpenACCNumWorkersClause>);
      if (Itr != ExistingClauses.end()) {
        SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
            << "num" << OpenACCClauseKind::Worker << Clause.getDirectiveKind()
            << HasAssocKind(Clause.getDirectiveKind(),
                            SemaRef.getActiveComputeConstructInfo().Kind)
            << SemaRef.getActiveComputeConstructInfo().Kind
            << OpenACCClauseKind::NumWorkers;
        SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
            << (*Itr)->getClauseKind();

        IntExpr = nullptr;
      }
    }
    }
  }

  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop) {
    // OpenACC 3.3 2.9.3: The region of a loop with a 'worker' clause may not
    // contain a loop with a gang or worker clause unless within a nested
    // compute region.
    if (SemaRef.LoopWorkerClauseLoc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Worker << OpenACCClauseKind::Worker
          << /*skip kernels construct info*/ 0;
      SemaRef.Diag(SemaRef.LoopWorkerClauseLoc,
                   diag::note_acc_previous_clause_here)
          << "worker";
      return nullptr;
    }

    // OpenACC 3.3 2.9.4: The region of a loop with a 'vector' clause may not
    // contain a loop with a gang, worker, or vector clause unless within a
    // nested compute region.
    if (SemaRef.LoopVectorClauseLoc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Worker << OpenACCClauseKind::Vector
          << /*skip kernels construct info*/ 0;
      SemaRef.Diag(SemaRef.LoopVectorClauseLoc,
                   diag::note_acc_previous_clause_here)
          << "vector";
      return nullptr;
    }
  }

  return OpenACCWorkerClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), IntExpr,
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitGangClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DiagGangWorkerVectorSeqConflict(Clause))
    return nullptr;

  // OpenACC 3.3 Section 2.9.11: A reduction clause may not appear on a loop
  // directive that has a gang clause and is within a compute construct that has
  // a num_gangs clause with more than one explicit argument.
  if ((Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop &&
       SemaRef.getActiveComputeConstructInfo().Kind !=
           OpenACCDirectiveKind::Invalid) ||
      isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
    // num_gangs clause on the active compute construct.
    auto ActiveComputeConstructContainer =
        isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())
            ? ExistingClauses
            : SemaRef.getActiveComputeConstructInfo().Clauses;
    auto *NumGangsClauseItr = llvm::find_if(
        ActiveComputeConstructContainer, llvm::IsaPred<OpenACCNumGangsClause>);

    if (NumGangsClauseItr != ActiveComputeConstructContainer.end() &&
        cast<OpenACCNumGangsClause>(*NumGangsClauseItr)->getIntExprs().size() >
            1) {
      auto *ReductionClauseItr =
          llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCReductionClause>);

      if (ReductionClauseItr != ExistingClauses.end()) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_gang_reduction_numgangs_conflict)
            << OpenACCClauseKind::Gang << OpenACCClauseKind::Reduction
            << Clause.getDirectiveKind()
            << isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind());
        SemaRef.Diag((*ReductionClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here)
            << (*ReductionClauseItr)->getClauseKind();
        SemaRef.Diag((*NumGangsClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here)
            << (*NumGangsClauseItr)->getClauseKind();
        return nullptr;
      }
    }
  }

  llvm::SmallVector<OpenACCGangKind> GangKinds;
  llvm::SmallVector<Expr *> IntExprs;

  // Store the existing locations, so we can do duplicate checking.  Index is
  // the int-value of the OpenACCGangKind enum.
  SourceLocation ExistingElemLoc[3];

  for (unsigned I = 0; I < Clause.getIntExprs().size(); ++I) {
    OpenACCGangKind GK = Clause.getGangKinds()[I];
    ExprResult ER =
        SemaRef.CheckGangExpr(ExistingClauses, Clause.getDirectiveKind(), GK,
                              Clause.getIntExprs()[I]);

    if (!ER.isUsable())
      continue;

    // OpenACC 3.3 2.9: 'gang-arg-list' may have at most one num, one dim, and
    // one static argument.
    if (ExistingElemLoc[static_cast<unsigned>(GK)].isValid()) {
      SemaRef.Diag(ER.get()->getBeginLoc(), diag::err_acc_gang_multiple_elt)
          << static_cast<unsigned>(GK);
      SemaRef.Diag(ExistingElemLoc[static_cast<unsigned>(GK)],
                   diag::note_acc_previous_expr_here);
      continue;
    }

    ExistingElemLoc[static_cast<unsigned>(GK)] = ER.get()->getBeginLoc();
    GangKinds.push_back(GK);
    IntExprs.push_back(ER.get());
  }

  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop) {
    // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
    // construct, the gang clause behaves as follows. ... The region of a loop
    // with a gang clause may not contain another loop with a gang clause unless
    // within a nested compute region.
    if (SemaRef.LoopGangClauseOnKernel.Loc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Gang << OpenACCClauseKind::Gang
          << /*kernels construct info*/ 1
          << SemaRef.LoopGangClauseOnKernel.DirKind;
      SemaRef.Diag(SemaRef.LoopGangClauseOnKernel.Loc,
                   diag::note_acc_previous_clause_here)
          << "gang";
      return nullptr;
    }

    // OpenACC 3.3 2.9.3: The region of a loop with a 'worker' clause may not
    // contain a loop with a gang or worker clause unless within a nested
    // compute region.
    if (SemaRef.LoopWorkerClauseLoc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Gang << OpenACCClauseKind::Worker
          << /*!kernels construct info*/ 0;
      SemaRef.Diag(SemaRef.LoopWorkerClauseLoc,
                   diag::note_acc_previous_clause_here)
          << "worker";
      return nullptr;
    }

    // OpenACC 3.3 2.9.4: The region of a loop with a 'vector' clause may not
    // contain a loop with a gang, worker, or vector clause unless within a
    // nested compute region.
    if (SemaRef.LoopVectorClauseLoc.isValid()) {
      // This handles the 'inner loop' diagnostic, but we cannot set that we're
      // on one of these until we get to the end of the construct.
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_in_clause_region)
          << OpenACCClauseKind::Gang << OpenACCClauseKind::Vector
          << /*!kernels construct info*/ 0;
      SemaRef.Diag(SemaRef.LoopVectorClauseLoc,
                   diag::note_acc_previous_clause_here)
          << "vector";
      return nullptr;
    }
  }

  return SemaRef.CheckGangClause(Clause.getDirectiveKind(), ExistingClauses,
                                 Clause.getBeginLoc(), Clause.getLParenLoc(),
                                 GangKinds, IntExprs, Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitFinalizeClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // There isn't anything to do here, this is only valid on one construct, and
  // has no associated rules.
  return OpenACCFinalizeClause::Create(Ctx, Clause.getBeginLoc(),
                                       Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitIfPresentClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // There isn't anything to do here, this is only valid on one construct, and
  // has no associated rules.
  return OpenACCIfPresentClause::Create(Ctx, Clause.getBeginLoc(),
                                        Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitSeqClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // OpenACC 3.3 2.9:
  //  A 'gang', 'worker', or 'vector' clause may not appear if a 'seq' clause
  //  appears.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop ||
      isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
    const auto *Itr = llvm::find_if(
        ExistingClauses, llvm::IsaPred<OpenACCGangClause, OpenACCVectorClause,
                                       OpenACCWorkerClause>);
    if (Itr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_cannot_combine)
          << Clause.getClauseKind() << (*Itr)->getClauseKind()
          << Clause.getDirectiveKind();
      SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here)
          << (*Itr)->getClauseKind();
      return nullptr;
    }
  }

  return OpenACCSeqClause::Create(Ctx, Clause.getBeginLoc(),
                                  Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitReductionClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // OpenACC 3.3 Section 2.9.11: A reduction clause may not appear on a loop
  // directive that has a gang clause and is within a compute construct that has
  // a num_gangs clause with more than one explicit argument.
  if ((Clause.getDirectiveKind() == OpenACCDirectiveKind::Loop &&
       SemaRef.getActiveComputeConstructInfo().Kind !=
           OpenACCDirectiveKind::Invalid) ||
      isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
    // num_gangs clause on the active compute construct.
    auto ActiveComputeConstructContainer =
        isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())
            ? ExistingClauses
            : SemaRef.getActiveComputeConstructInfo().Clauses;
    auto *NumGangsClauseItr = llvm::find_if(
        ActiveComputeConstructContainer, llvm::IsaPred<OpenACCNumGangsClause>);

    if (NumGangsClauseItr != ActiveComputeConstructContainer.end() &&
        cast<OpenACCNumGangsClause>(*NumGangsClauseItr)->getIntExprs().size() >
            1) {
      auto *GangClauseItr =
          llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCGangClause>);

      if (GangClauseItr != ExistingClauses.end()) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_gang_reduction_numgangs_conflict)
            << OpenACCClauseKind::Reduction << OpenACCClauseKind::Gang
            << Clause.getDirectiveKind()
            << isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind());
        SemaRef.Diag((*GangClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here)
            << (*GangClauseItr)->getClauseKind();
        SemaRef.Diag((*NumGangsClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here)
            << (*NumGangsClauseItr)->getClauseKind();
        return nullptr;
      }
    }
  }

  // OpenACC3.3 Section 2.9.11: If a variable is involved in a reduction that
  // spans multiple nested loops where two or more of those loops have
  // associated loop directives, a reduction clause containing that variable
  // must appear on each of those loop directives.
  //
  // This can't really be implemented in the CFE, as this requires a level of
  // rechability/useage analysis that we're not really wanting to get into.
  // Additionally, I'm alerted that this restriction is one that the middle-end
  // can just 'figure out' as an extension and isn't really necessary.
  //
  // OpenACC3.3 Section 2.9.11: Every 'var' in a reduction clause appearing on
  // an orphaned loop construct must be private.
  //
  // This again is something we cannot really diagnose, as it requires we see
  // all the uses/scopes of all variables referenced.  The middle end/MLIR might
  // be able to diagnose this.

  // OpenACC 3.3 Section 2.5.4:
  // A reduction clause may not appear on a parallel construct with a
  // num_gangs clause that has more than one argument.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Parallel ||
      Clause.getDirectiveKind() == OpenACCDirectiveKind::ParallelLoop) {
    auto NumGangsClauses = llvm::make_filter_range(
        ExistingClauses, llvm::IsaPred<OpenACCNumGangsClause>);

    for (auto *NGC : NumGangsClauses) {
      unsigned NumExprs =
          cast<OpenACCNumGangsClause>(NGC)->getIntExprs().size();

      if (NumExprs > 1) {
        SemaRef.Diag(Clause.getBeginLoc(),
                     diag::err_acc_reduction_num_gangs_conflict)
            << /*>1 arg in first loc=*/0 << Clause.getClauseKind()
            << Clause.getDirectiveKind() << OpenACCClauseKind::NumGangs;
        SemaRef.Diag(NGC->getBeginLoc(), diag::note_acc_previous_clause_here)
            << NGC->getClauseKind();
        return nullptr;
      }
    }
  }

  SmallVector<Expr *> ValidVars;

  for (Expr *Var : Clause.getVarList()) {
    ExprResult Res = SemaRef.CheckReductionVar(Clause.getDirectiveKind(),
                                               Clause.getReductionOp(), Var);

    if (Res.isUsable())
      ValidVars.push_back(Res.get());
  }

  return SemaRef.CheckReductionClause(
      ExistingClauses, Clause.getDirectiveKind(), Clause.getBeginLoc(),
      Clause.getLParenLoc(), Clause.getReductionOp(), ValidVars,
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCollapseClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (DisallowSinceLastDeviceType<OpenACCCollapseClause>(Clause))
    return nullptr;

  ExprResult LoopCount = SemaRef.CheckCollapseLoopCount(Clause.getLoopCount());

  if (!LoopCount.isUsable())
    return nullptr;

  return OpenACCCollapseClause::Create(Ctx, Clause.getBeginLoc(),
                                       Clause.getLParenLoc(), Clause.isForce(),
                                       LoopCount.get(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitBindClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  if (std::holds_alternative<StringLiteral *>(Clause.getBindDetails()))
    return OpenACCBindClause::Create(
        Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(),
        std::get<StringLiteral *>(Clause.getBindDetails()), Clause.getEndLoc());
  return OpenACCBindClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(),
      std::get<IdentifierInfo *>(Clause.getBindDetails()), Clause.getEndLoc());
}

// Return true if the two vars refer to the same variable, for the purposes of
// equality checking.
bool areVarsEqual(Expr *VarExpr1, Expr *VarExpr2) {
  if (VarExpr1->isInstantiationDependent() ||
      VarExpr2->isInstantiationDependent())
    return false;

  VarExpr1 = VarExpr1->IgnoreParenCasts();
  VarExpr2 = VarExpr2->IgnoreParenCasts();

  // Legal expressions can be: Scalar variable reference, sub-array, array
  // element, or composite variable member.

  // Sub-array.
  if (isa<ArraySectionExpr>(VarExpr1)) {
    auto *Expr2AS = dyn_cast<ArraySectionExpr>(VarExpr2);
    if (!Expr2AS)
      return false;

    auto *Expr1AS = cast<ArraySectionExpr>(VarExpr1);

    if (!areVarsEqual(Expr1AS->getBase(), Expr2AS->getBase()))
      return false;
    // We could possibly check to see if the ranges aren't overlapping, but it
    // isn't clear that the rules allow this.
    return true;
  }

  // Array-element.
  if (isa<ArraySubscriptExpr>(VarExpr1)) {
    auto *Expr2AS = dyn_cast<ArraySubscriptExpr>(VarExpr2);
    if (!Expr2AS)
      return false;

    auto *Expr1AS = cast<ArraySubscriptExpr>(VarExpr1);

    if (!areVarsEqual(Expr1AS->getBase(), Expr2AS->getBase()))
      return false;

    // We could possibly check to see if the elements referenced aren't the
    // same, but it isn't clear by reading of the standard that this is allowed
    // (and that the 'var' refered to isn't the array).
    return true;
  }

  // Scalar variable reference, or composite variable.
  if (isa<DeclRefExpr>(VarExpr1)) {
    auto *Expr2DRE = dyn_cast<DeclRefExpr>(VarExpr2);
    if (!Expr2DRE)
      return false;

    auto *Expr1DRE = cast<DeclRefExpr>(VarExpr1);

    return Expr1DRE->getDecl()->getMostRecentDecl() ==
           Expr2DRE->getDecl()->getMostRecentDecl();
  }

  llvm_unreachable("Unknown variable type encountered");
}
} // namespace

OpenACCClause *
SemaOpenACC::ActOnClause(ArrayRef<const OpenACCClause *> ExistingClauses,
                         OpenACCParsedClause &Clause) {
  if (Clause.getClauseKind() == OpenACCClauseKind::Invalid)
    return nullptr;

  if (DiagnoseAllowedClauses(Clause.getDirectiveKind(), Clause.getClauseKind(),
                             Clause.getBeginLoc()))
    return nullptr;
  //// Diagnose that we don't support this clause on this directive.
  // if (!doesClauseApplyToDirective(Clause.getDirectiveKind(),
  //                                 Clause.getClauseKind())) {
  //   Diag(Clause.getBeginLoc(), diag::err_acc_clause_appertainment)
  //       << Clause.getDirectiveKind() << Clause.getClauseKind();
  //   return nullptr;
  // }

  if (const auto *DevTypeClause = llvm::find_if(
          ExistingClauses, llvm::IsaPred<OpenACCDeviceTypeClause>);
      DevTypeClause != ExistingClauses.end()) {
    if (checkValidAfterDeviceType(
            *this, *cast<OpenACCDeviceTypeClause>(*DevTypeClause), Clause))
      return nullptr;
  }

  SemaOpenACCClauseVisitor Visitor{*this, ExistingClauses};
  OpenACCClause *Result = Visitor.Visit(Clause);
  assert((!Result || Result->getClauseKind() == Clause.getClauseKind()) &&
         "Created wrong clause?");

  return Result;
}

/// OpenACC 3.3 section 2.5.15:
/// At a mininmum, the supported data types include ... the numerical data types
/// in C, C++, and Fortran.
///
/// If the reduction var is a composite variable, each
/// member of the composite variable must be a supported datatype for the
/// reduction operation.
ExprResult SemaOpenACC::CheckReductionVar(OpenACCDirectiveKind DirectiveKind,
                                          OpenACCReductionOperator ReductionOp,
                                          Expr *VarExpr) {
  VarExpr = VarExpr->IgnoreParenCasts();

  auto TypeIsValid = [](QualType Ty) {
    return Ty->isDependentType() || Ty->isScalarType();
  };

  if (isa<ArraySectionExpr>(VarExpr)) {
    Expr *ASExpr = VarExpr;
    QualType BaseTy = ArraySectionExpr::getBaseOriginalType(ASExpr);
    QualType EltTy = getASTContext().getBaseElementType(BaseTy);

    if (!TypeIsValid(EltTy)) {
      Diag(VarExpr->getExprLoc(), diag::err_acc_reduction_type)
          << EltTy << /*Sub array base type*/ 1;
      return ExprError();
    }
  } else if (VarExpr->getType()->isArrayType()) {
    // Arrays are considered an 'aggregate variable' explicitly, so are OK, no
    // additional checking required.
    //
    // Glossary: Aggregate variables – a variable of any non-scalar datatype,
    // including array or composite variables.
    //
    // The next branch (record decl) checks for composite variables.
  } else if (auto *RD = VarExpr->getType()->getAsRecordDecl()) {
    if (!RD->isStruct() && !RD->isClass()) {
      Diag(VarExpr->getExprLoc(), diag::err_acc_reduction_composite_type)
          << /*not class or struct*/ 0 << VarExpr->getType();
      return ExprError();
    }

    if (!RD->isCompleteDefinition()) {
      Diag(VarExpr->getExprLoc(), diag::err_acc_reduction_composite_type)
          << /*incomplete*/ 1 << VarExpr->getType();
      return ExprError();
    }
    if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD);
        CXXRD && !CXXRD->isAggregate()) {
      Diag(VarExpr->getExprLoc(), diag::err_acc_reduction_composite_type)
          << /*aggregate*/ 2 << VarExpr->getType();
      return ExprError();
    }

    for (FieldDecl *FD : RD->fields()) {
      if (!TypeIsValid(FD->getType())) {
        Diag(VarExpr->getExprLoc(),
             diag::err_acc_reduction_composite_member_type);
        Diag(FD->getLocation(), diag::note_acc_reduction_composite_member_loc);
        return ExprError();
      }
    }
  } else if (!TypeIsValid(VarExpr->getType())) {
    Diag(VarExpr->getExprLoc(), diag::err_acc_reduction_type)
        << VarExpr->getType() << /*Sub array base type*/ 0;
    return ExprError();
  }

  // OpenACC3.3: 2.9.11: Reduction clauses on nested constructs for the same
  // reduction 'var' must have the same reduction operator.
  if (!VarExpr->isInstantiationDependent()) {

    for (const OpenACCReductionClause *RClause : ActiveReductionClauses) {
      if (RClause->getReductionOp() == ReductionOp)
        break;

      for (Expr *OldVarExpr : RClause->getVarList()) {
        if (OldVarExpr->isInstantiationDependent())
          continue;

        if (areVarsEqual(VarExpr, OldVarExpr)) {
          Diag(VarExpr->getExprLoc(), diag::err_reduction_op_mismatch)
              << ReductionOp << RClause->getReductionOp();
          Diag(OldVarExpr->getExprLoc(), diag::note_acc_previous_clause_here)
              << RClause->getClauseKind();
          return ExprError();
        }
      }
    }
  }

  return VarExpr;
}

ExprResult SemaOpenACC::CheckTileSizeExpr(Expr *SizeExpr) {
  if (!SizeExpr)
    return ExprError();

  assert((SizeExpr->isInstantiationDependent() ||
          SizeExpr->getType()->isIntegerType()) &&
         "size argument non integer?");

  // If dependent, or an asterisk, the expression is fine.
  if (SizeExpr->isInstantiationDependent() ||
      isa<OpenACCAsteriskSizeExpr>(SizeExpr))
    return ExprResult{SizeExpr};

  std::optional<llvm::APSInt> ICE =
      SizeExpr->getIntegerConstantExpr(getASTContext());

  // OpenACC 3.3 2.9.8
  // where each tile size is a constant positive integer expression or asterisk.
  if (!ICE || *ICE <= 0) {
    Diag(SizeExpr->getBeginLoc(), diag::err_acc_size_expr_value)
        << ICE.has_value() << ICE.value_or(llvm::APSInt{}).getExtValue();
    return ExprError();
  }

  return ExprResult{
      ConstantExpr::Create(getASTContext(), SizeExpr, APValue{*ICE})};
}

ExprResult SemaOpenACC::CheckCollapseLoopCount(Expr *LoopCount) {
  if (!LoopCount)
    return ExprError();

  assert((LoopCount->isInstantiationDependent() ||
          LoopCount->getType()->isIntegerType()) &&
         "Loop argument non integer?");

  // If this is dependent, there really isn't anything we can check.
  if (LoopCount->isInstantiationDependent())
    return ExprResult{LoopCount};

  std::optional<llvm::APSInt> ICE =
      LoopCount->getIntegerConstantExpr(getASTContext());

  // OpenACC 3.3: 2.9.1
  // The argument to the collapse clause must be a constant positive integer
  // expression.
  if (!ICE || *ICE <= 0) {
    Diag(LoopCount->getBeginLoc(), diag::err_acc_collapse_loop_count)
        << ICE.has_value() << ICE.value_or(llvm::APSInt{}).getExtValue();
    return ExprError();
  }

  return ExprResult{
      ConstantExpr::Create(getASTContext(), LoopCount, APValue{*ICE})};
}

ExprResult
SemaOpenACC::CheckGangExpr(ArrayRef<const OpenACCClause *> ExistingClauses,
                           OpenACCDirectiveKind DK, OpenACCGangKind GK,
                           Expr *E) {
  // There are two cases for the enforcement here: the 'current' directive is a
  // 'loop', where we need to check the active compute construct kind, or the
  // current directive is a 'combined' construct, where we have to check the
  // current one.
  switch (DK) {
  case OpenACCDirectiveKind::ParallelLoop:
    return CheckGangParallelExpr(*this, DK, ActiveComputeConstructInfo.Kind, GK,
                                 E);
  case OpenACCDirectiveKind::SerialLoop:
    return CheckGangSerialExpr(*this, DK, ActiveComputeConstructInfo.Kind, GK,
                               E);
  case OpenACCDirectiveKind::KernelsLoop:
    return CheckGangKernelsExpr(*this, ExistingClauses, DK,
                                ActiveComputeConstructInfo.Kind, GK, E);
  case OpenACCDirectiveKind::Routine:
    return CheckGangRoutineExpr(*this, DK, ActiveComputeConstructInfo.Kind, GK,
                                E);
  case OpenACCDirectiveKind::Loop:
    switch (ActiveComputeConstructInfo.Kind) {
    case OpenACCDirectiveKind::Invalid:
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::ParallelLoop:
      return CheckGangParallelExpr(*this, DK, ActiveComputeConstructInfo.Kind,
                                   GK, E);
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::Serial:
      return CheckGangSerialExpr(*this, DK, ActiveComputeConstructInfo.Kind, GK,
                                 E);
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Kernels:
      return CheckGangKernelsExpr(*this, ExistingClauses, DK,
                                  ActiveComputeConstructInfo.Kind, GK, E);
    default:
      llvm_unreachable("Non compute construct in active compute construct?");
    }
  case OpenACCDirectiveKind::Invalid:
    // This can happen in cases where the the directive was not recognized but
    // we continued anyway. Since the validity checking is all-over the place
    // (it can be a star/integer, or a constant expr depending on the tag), we
    // just give up and return an ExprError here.
    return ExprError();
  default:
    llvm_unreachable("Invalid directive kind for a Gang clause");
  }
  llvm_unreachable("Compute construct directive not handled?");
}

OpenACCClause *
SemaOpenACC::CheckGangClause(OpenACCDirectiveKind DirKind,
                             ArrayRef<const OpenACCClause *> ExistingClauses,
                             SourceLocation BeginLoc, SourceLocation LParenLoc,
                             ArrayRef<OpenACCGangKind> GangKinds,
                             ArrayRef<Expr *> IntExprs, SourceLocation EndLoc) {
  // Reduction isn't possible on 'routine' so we don't bother checking it here.
  if (DirKind != OpenACCDirectiveKind::Routine) {
    // OpenACC 3.3 2.9.11: A reduction clause may not appear on a loop directive
    // that has a gang clause with a dim: argument whose value is greater
    // than 1.
    const auto *ReductionItr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCReductionClause>);

    if (ReductionItr != ExistingClauses.end()) {
      const auto GangZip = llvm::zip_equal(GangKinds, IntExprs);
      const auto GangItr = llvm::find_if(GangZip, [](const auto &Tuple) {
        return std::get<0>(Tuple) == OpenACCGangKind::Dim;
      });

      if (GangItr != GangZip.end()) {
        const Expr *DimExpr = std::get<1>(*GangItr);

        assert((DimExpr->isInstantiationDependent() ||
                isa<ConstantExpr>(DimExpr)) &&
               "Improperly formed gang argument");
        if (const auto *DimVal = dyn_cast<ConstantExpr>(DimExpr);
            DimVal && DimVal->getResultAsAPSInt() > 1) {
          Diag(DimVal->getBeginLoc(), diag::err_acc_gang_reduction_conflict)
              << /*gang/reduction=*/0 << DirKind;
          Diag((*ReductionItr)->getBeginLoc(),
               diag::note_acc_previous_clause_here)
              << (*ReductionItr)->getClauseKind();
          return nullptr;
        }
      }
    }
  }

  return OpenACCGangClause::Create(getASTContext(), BeginLoc, LParenLoc,
                                   GangKinds, IntExprs, EndLoc);
}

OpenACCClause *SemaOpenACC::CheckReductionClause(
    ArrayRef<const OpenACCClause *> ExistingClauses,
    OpenACCDirectiveKind DirectiveKind, SourceLocation BeginLoc,
    SourceLocation LParenLoc, OpenACCReductionOperator ReductionOp,
    ArrayRef<Expr *> Vars, SourceLocation EndLoc) {
  if (DirectiveKind == OpenACCDirectiveKind::Loop ||
      isOpenACCCombinedDirectiveKind(DirectiveKind)) {
    // OpenACC 3.3 2.9.11: A reduction clause may not appear on a loop directive
    // that has a gang clause with a dim: argument whose value is greater
    // than 1.
    const auto GangClauses = llvm::make_filter_range(
        ExistingClauses, llvm::IsaPred<OpenACCGangClause>);

    for (auto *GC : GangClauses) {
      const auto *GangClause = cast<OpenACCGangClause>(GC);
      for (unsigned I = 0; I < GangClause->getNumExprs(); ++I) {
        std::pair<OpenACCGangKind, const Expr *> EPair = GangClause->getExpr(I);
        if (EPair.first != OpenACCGangKind::Dim)
          continue;

        if (const auto *DimVal = dyn_cast<ConstantExpr>(EPair.second);
            DimVal && DimVal->getResultAsAPSInt() > 1) {
          Diag(BeginLoc, diag::err_acc_gang_reduction_conflict)
              << /*reduction/gang=*/1 << DirectiveKind;
          Diag(GangClause->getBeginLoc(), diag::note_acc_previous_clause_here)
              << GangClause->getClauseKind();
          return nullptr;
        }
      }
    }
  }

  auto *Ret = OpenACCReductionClause::Create(
      getASTContext(), BeginLoc, LParenLoc, ReductionOp, Vars, EndLoc);
  return Ret;
}

llvm::SmallVector<Expr *>
SemaOpenACC::CheckLinkClauseVarList(ArrayRef<Expr *> VarExprs) {
  const DeclContext *DC = removeLinkageSpecDC(getCurContext());

  // Link has no special restrictions on its var list unless it is not at NS/TU
  // scope.
  if (isa<NamespaceDecl, TranslationUnitDecl>(DC))
    return llvm::SmallVector<Expr *>(VarExprs);

  llvm::SmallVector<Expr *> NewVarList;

  for (Expr *VarExpr : VarExprs) {
    if (isa<DependentScopeDeclRefExpr, CXXDependentScopeMemberExpr>(VarExpr)) {
      NewVarList.push_back(VarExpr);
      continue;
    }

    // Field decls can't be global, nor extern, and declare can't refer to
    // non-static fields in class-scope, so this always fails the scope check.
    // BUT for now we add this so it gets diagnosed by the general 'declare'
    // rules.
    if (isa<MemberExpr>(VarExpr)) {
      NewVarList.push_back(VarExpr);
      continue;
    }

    const auto *DRE = cast<DeclRefExpr>(VarExpr);
    const VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl());

    if (!Var || !Var->hasExternalStorage())
      Diag(VarExpr->getBeginLoc(), diag::err_acc_link_not_extern);
    else
      NewVarList.push_back(VarExpr);
  }

  return NewVarList;
}
bool SemaOpenACC::CheckDeclareClause(SemaOpenACC::OpenACCParsedClause &Clause,
                                     OpenACCModifierKind Mods) {

  if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Declare)
    return false;

  const DeclContext *DC = removeLinkageSpecDC(getCurContext());

  // Whether this is 'create', 'copyin', 'deviceptr', 'device_resident', or
  // 'link', which have 2 special rules.
  bool IsSpecialClause =
      Clause.getClauseKind() == OpenACCClauseKind::Create ||
      Clause.getClauseKind() == OpenACCClauseKind::CopyIn ||
      Clause.getClauseKind() == OpenACCClauseKind::DevicePtr ||
      Clause.getClauseKind() == OpenACCClauseKind::DeviceResident ||
      Clause.getClauseKind() == OpenACCClauseKind::Link;

  // OpenACC 3.3 2.13:
  // In C or C++ global or namespace scope, only 'create',
  // 'copyin', 'deviceptr', 'device_resident', or 'link' clauses are
  // allowed.
  if (!IsSpecialClause && isa<NamespaceDecl, TranslationUnitDecl>(DC)) {
    return Diag(Clause.getBeginLoc(), diag::err_acc_declare_clause_at_global)
           << Clause.getClauseKind();
  }

  llvm::SmallVector<Expr *> FilteredVarList;
  const DeclaratorDecl *CurDecl = nullptr;
  for (Expr *VarExpr : Clause.getVarList()) {
    if (isa<DependentScopeDeclRefExpr, CXXDependentScopeMemberExpr>(VarExpr)) {
      // There isn't really anything we can do here, so we add them anyway and
      // we can check them again when we instantiate this.
    } else if (const auto *MemExpr = dyn_cast<MemberExpr>(VarExpr)) {
      FieldDecl *FD =
          cast<FieldDecl>(MemExpr->getMemberDecl()->getCanonicalDecl());
      CurDecl = FD;

      if (removeLinkageSpecDC(
              FD->getLexicalDeclContext()->getPrimaryContext()) != DC) {
        Diag(MemExpr->getBeginLoc(), diag::err_acc_declare_same_scope)
            << Clause.getClauseKind();
        continue;
      }
    } else {

      const Expr *VarExprTemp = VarExpr;

      while (const auto *ASE = dyn_cast<ArraySectionExpr>(VarExprTemp))
        VarExprTemp = ASE->getBase()->IgnoreParenImpCasts();

      const auto *DRE = cast<DeclRefExpr>(VarExprTemp);
      if (const auto *Var = dyn_cast<VarDecl>(DRE->getDecl())) {
        CurDecl = Var->getCanonicalDecl();

        // OpenACC3.3 2.13:
        // A 'declare' directive must be in the same scope as the declaration of
        // any var that appears in the clauses of the directive or any scope
        // within a C/C++ function.
        // We can't really check 'scope' here, so we check declaration context,
        // which is a reasonable approximation, but misses scopes inside of
        // functions.
        if (removeLinkageSpecDC(
                Var->getLexicalDeclContext()->getPrimaryContext()) != DC) {
          Diag(VarExpr->getBeginLoc(), diag::err_acc_declare_same_scope)
              << Clause.getClauseKind();
          continue;
        }
        // OpenACC3.3 2.13:
        // C and C++ extern variables may only appear in 'create',
        // 'copyin', 'deviceptr', 'device_resident', or 'link' clauses on a
        // 'declare' directive.
        if (!IsSpecialClause && Var->hasExternalStorage()) {
          Diag(VarExpr->getBeginLoc(), diag::err_acc_declare_extern)
              << Clause.getClauseKind();
          continue;
        }
      }

      // OpenACC3.3 2.13:
      // A var may appear at most once in all the clauses of declare
      // directives for a function, subroutine, program, or module.

      if (CurDecl) {
        auto [Itr, Inserted] = DeclareVarReferences.try_emplace(CurDecl);
        if (!Inserted) {
          Diag(VarExpr->getBeginLoc(), diag::err_acc_multiple_references)
              << Clause.getClauseKind();
          Diag(Itr->second, diag::note_acc_previous_reference);
          continue;
        } else {
          Itr->second = VarExpr->getBeginLoc();
        }
      }
    }
    FilteredVarList.push_back(VarExpr);
  }

  Clause.setVarListDetails(FilteredVarList, Mods);
  return false;
}
