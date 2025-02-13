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

#include "clang/AST/OpenACCClause.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/SemaOpenACC.h"

using namespace clang;

namespace {
bool doesClauseApplyToDirective(OpenACCDirectiveKind DirectiveKind,
                                OpenACCClauseKind ClauseKind) {
  switch (ClauseKind) {
    // FIXME: For each clause as we implement them, we can add the
    // 'legalization' list here.
  case OpenACCClauseKind::Default:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Data:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::If:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ExitData:
    case OpenACCDirectiveKind::HostData:
    case OpenACCDirectiveKind::Init:
    case OpenACCDirectiveKind::Shutdown:
    case OpenACCDirectiveKind::Set:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::Wait:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Self:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::NumGangs:
  case OpenACCClauseKind::NumWorkers:
  case OpenACCClauseKind::VectorLength:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::FirstPrivate:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Private:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::NoCreate:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Present:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::Declare:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Copy:
  case OpenACCClauseKind::PCopy:
  case OpenACCClauseKind::PresentOrCopy:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::Declare:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::CopyIn:
  case OpenACCClauseKind::PCopyIn:
  case OpenACCClauseKind::PresentOrCopyIn:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::Declare:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::CopyOut:
  case OpenACCClauseKind::PCopyOut:
  case OpenACCClauseKind::PresentOrCopyOut:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::ExitData:
    case OpenACCDirectiveKind::Declare:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Create:
  case OpenACCClauseKind::PCreate:
  case OpenACCClauseKind::PresentOrCreate:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Attach:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::DevicePtr:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::Declare:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Async:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ExitData:
    case OpenACCDirectiveKind::Set:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::Wait:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Wait:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ExitData:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Seq:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::Routine:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Independent:
  case OpenACCClauseKind::Auto:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Reduction:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::DeviceType:
  case OpenACCClauseKind::DType:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::Init:
    case OpenACCDirectiveKind::Shutdown:
    case OpenACCDirectiveKind::Set:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::Routine:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }

  case OpenACCClauseKind::Collapse: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Tile: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  }

  case OpenACCClauseKind::Gang: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Routine:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Worker: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Routine:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Vector: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Loop:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Routine:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Finalize: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::ExitData:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::IfPresent: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::HostData:
    case OpenACCDirectiveKind::Update:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Delete: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::ExitData:
      return true;
    default:
      return false;
    }
  }

  case OpenACCClauseKind::Detach: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::ExitData:
      return true;
    default:
      return false;
    }
  }

  case OpenACCClauseKind::DeviceNum: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Init:
    case OpenACCDirectiveKind::Shutdown:
    case OpenACCDirectiveKind::Set:
      return true;
    default:
      return false;
    }
  }

  case OpenACCClauseKind::UseDevice: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::HostData:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::DefaultAsync: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Set:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Device: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Update:
      return true;
    default:
      return false;
    }
  }
  case OpenACCClauseKind::Host: {
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Update:
      return true;
    default:
      return false;
    }
  }
  }

  default:
    // Do nothing so we can go to the 'unimplemented' diagnostic instead.
    return true;
  }
  llvm_unreachable("Invalid clause kind");
}

bool checkAlreadyHasClauseOfKind(
    SemaOpenACC &S, ArrayRef<const OpenACCClause *> ExistingClauses,
    SemaOpenACC::OpenACCParsedClause &Clause) {
  const auto *Itr = llvm::find_if(ExistingClauses, [&](const OpenACCClause *C) {
    return C->getClauseKind() == Clause.getClauseKind();
  });
  if (Itr != ExistingClauses.end()) {
    S.Diag(Clause.getBeginLoc(), diag::err_acc_duplicate_clause_disallowed)
        << Clause.getDirectiveKind() << Clause.getClauseKind();
    S.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return true;
  }
  return false;
}
bool checkValidAfterDeviceType(
    SemaOpenACC &S, const OpenACCDeviceTypeClause &DeviceTypeClause,
    const SemaOpenACC::OpenACCParsedClause &NewClause) {
  // This is implemented for everything but 'routine', so treat as 'fine' for
  // that.
  if (NewClause.getDirectiveKind() == OpenACCDirectiveKind::Routine)
    return false;

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
  }
  S.Diag(NewClause.getBeginLoc(), diag::err_acc_clause_after_device_type)
      << NewClause.getClauseKind() << DeviceTypeClause.getClauseKind()
      << NewClause.getDirectiveKind();
  S.Diag(DeviceTypeClause.getBeginLoc(), diag::note_acc_previous_clause_here);
  return true;
}

// A temporary function that helps implement the 'not implemented' check at the
// top of each clause checking function. This should only be used in conjunction
// with the one being currently implemented/only updated after the entire
// construct has been implemented.
bool isDirectiveKindImplemented(OpenACCDirectiveKind DK) {
  return DK != OpenACCDirectiveKind::Declare &&
         DK != OpenACCDirectiveKind::Routine;
}

class SemaOpenACCClauseVisitor {
  SemaOpenACC &SemaRef;
  ASTContext &Ctx;
  ArrayRef<const OpenACCClause *> ExistingClauses;
  bool NotImplemented = false;

  OpenACCClause *isNotImplemented() {
    NotImplemented = true;
    return nullptr;
  }

  // OpenACC 3.3 2.9:
  // A 'gang', 'worker', or 'vector' clause may not appear if a 'seq' clause
  // appears.
  bool DiagIfSeqClause(SemaOpenACC::OpenACCParsedClause &Clause) {
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCSeqClause>);

    if (Itr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_cannot_combine)
          << Clause.getClauseKind() << (*Itr)->getClauseKind()
          << Clause.getDirectiveKind();
      SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);

      return true;
    }
    return false;
  }

public:
  SemaOpenACCClauseVisitor(SemaOpenACC &S,
                           ArrayRef<const OpenACCClause *> ExistingClauses)
      : SemaRef(S), Ctx(S.getASTContext()), ExistingClauses(ExistingClauses) {}
  // Once we've implemented everything, we shouldn't need this infrastructure.
  // But in the meantime, we use this to help decide whether the clause was
  // handled for this directive.
  bool diagNotImplemented() { return NotImplemented; }

  OpenACCClause *Visit(SemaOpenACC::OpenACCParsedClause &Clause) {
    switch (Clause.getClauseKind()) {
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
    default:
      return isNotImplemented();
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

  // OpenACC 3.3, Section 2.5.4:
  // At most one 'default' clause may appear, and it must have a value of
  // either 'none' or 'present'.
  // Second half of the sentence is diagnosed during parsing.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

  return OpenACCDefaultClause::Create(
      Ctx, Clause.getDefaultClauseKind(), Clause.getBeginLoc(),
      Clause.getLParenLoc(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitTileClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {

  // Duplicates here are not really sensible.  We could possible permit
  // multiples if they all had the same value, but there isn't really a good
  // reason to do so. Also, this simplifies the suppression of duplicates, in
  // that we know if we 'find' one after instantiation, that it is the same
  // clause, which simplifies instantiation/checking/etc.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
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
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense. Prose DOES exist for 'data' and 'host_data', 'set', 'enter data' and
  // 'exit data' both don't, but other implmementations do this.  OpenACC issue
  // 519 filed for the latter two. Prose also exists for 'update'.
  // GCC allows this on init/shutdown, presumably for good reason, so we do too.
  if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Init &&
      Clause.getDirectiveKind() != OpenACCDirectiveKind::Shutdown &&
      checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

  // The parser has ensured that we have a proper condition expr, so there
  // isn't really much to do here.

  // If the 'if' clause is true, it makes the 'self' clause have no effect,
  // diagnose that here.  This only applies on compute/combined constructs.
  if (Clause.getDirectiveKind() != OpenACCDirectiveKind::Update) {
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCSelfClause>);
    if (Itr != ExistingClauses.end()) {
      SemaRef.Diag(Clause.getBeginLoc(), diag::warn_acc_if_self_conflict);
      SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    }
  }

  return OpenACCIfClause::Create(Ctx, Clause.getBeginLoc(),
                                 Clause.getLParenLoc(),
                                 Clause.getConditionExpr(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitSelfClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

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
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
  }
  return OpenACCSelfClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getConditionExpr(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitNumGangsClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
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
                   diag::note_acc_previous_clause_here);
      SemaRef.Diag((*GangClauseItr)->getBeginLoc(),
                   diag::note_acc_previous_clause_here);
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
                   diag::note_acc_previous_clause_here);
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
        SemaRef.Diag(GC->getBeginLoc(), diag::note_acc_previous_clause_here);
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
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
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
        SemaRef.Diag(WC->getBeginLoc(), diag::note_acc_previous_clause_here);
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
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
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
        SemaRef.Diag(VC->getBeginLoc(), diag::note_acc_previous_clause_here);
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
  // There is no prose in the standard that says duplicates aren't allowed,
  // but this diagnostic is present in other compilers, as well as makes
  // sense.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
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
  // Restrictions only properly implemented on certain constructs, so skip/treat
  // as unimplemented in those cases.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

  // OpenACC 3.3 2.14.3: Two instances of the same clause may not appear on the
  // same directive.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Set &&
      checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

  assert(Clause.getNumIntExprs() == 1 &&
         "Invalid number of expressions for device_num");
  return OpenACCDeviceNumClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getIntExprs()[0],
      Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitDefaultAsyncClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // OpenACC 3.3 2.14.3: Two instances of the same clause may not appear on the
  // same directive.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

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

  return OpenACCPrivateClause::Create(Ctx, Clause.getBeginLoc(),
                                      Clause.getLParenLoc(),
                                      Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitFirstPrivateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCFirstPrivateClause::Create(
      Ctx, Clause.getBeginLoc(), Clause.getLParenLoc(), Clause.getVarList(),
      Clause.getEndLoc());
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
  // Restrictions only properly implemented on 'compute'/'combined'/'data'
  // constructs, and 'compute'/'combined'/'data' constructs are the only
  // construct that can do anything with this yet, so skip/treat as
  // unimplemented in this case.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

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
  // Restrictions only properly implemented on 'compute'/'combined'/'data'
  // constructs, and 'compute'/'combined'/'data' constructs are the only
  // construct that can do anything with this yet, so skip/treat as
  // unimplemented in this case.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCCopyClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCopyInClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // Restrictions only properly implemented on 'compute'/'combined'/'data'
  // constructs, and 'compute'/'combined'/'data' constructs are the only
  // construct that can do anything with this yet, so skip/treat as
  // unimplemented in this case.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCCopyInClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.isReadOnly(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCopyOutClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // Restrictions only properly implemented on 'compute'/'combined'/'data'
  // constructs, and 'compute'/'combined'/'data' constructs are the only
  // construct that can do anything with this yet, so skip/treat as
  // unimplemented in this case.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCCopyOutClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.isZero(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitCreateClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, so there
  // really isn't anything to do here. GCC does some duplicate-finding, though
  // it isn't apparent in the standard where this is justified.

  return OpenACCCreateClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.isZero(), Clause.getVarList(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitAttachClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // ActOnVar ensured that everything is a valid variable reference, but we
  // still have to make sure it is a pointer type.
  llvm::SmallVector<Expr *> VarList{Clause.getVarList()};
  llvm::erase_if(VarList, [&](Expr *E) {
    return SemaRef.CheckVarIsPointerType(OpenACCClauseKind::Attach, E);
  });
  Clause.setVarListDetails(VarList,
                           /*IsReadOnly=*/false, /*IsZero=*/false);
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
  Clause.setVarListDetails(VarList,
                           /*IsReadOnly=*/false, /*IsZero=*/false);
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
  // Restrictions only properly implemented on 'compute'/'combined'/'data'
  // constructs, and 'compute'/'combined'/'data' constructs are the only
  // construct that can do anything with this yet, so skip/treat as
  // unimplemented in this case.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

  // ActOnVar ensured that everything is a valid variable reference, but we
  // still have to make sure it is a pointer type.
  llvm::SmallVector<Expr *> VarList{Clause.getVarList()};
  llvm::erase_if(VarList, [&](Expr *E) {
    return SemaRef.CheckVarIsPointerType(OpenACCClauseKind::DevicePtr, E);
  });
  Clause.setVarListDetails(VarList,
                           /*IsReadOnly=*/false, /*IsZero=*/false);

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
  // Restrictions implemented properly on everything except 'routine'.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Routine)
    return isNotImplemented();

  // OpenACC 3.3 2.14.3: Two instances of the same clause may not appear on the
  // same directive.
  if (Clause.getDirectiveKind() == OpenACCDirectiveKind::Set &&
      checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

  // TODO OpenACC: Once we get enough of the CodeGen implemented that we have
  // a source for the list of valid architectures, we need to warn on unknown
  // identifiers here.

  return OpenACCDeviceTypeClause::Create(
      Ctx, Clause.getClauseKind(), Clause.getBeginLoc(), Clause.getLParenLoc(),
      Clause.getDeviceTypeArchitectures(), Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitAutoClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // OpenACC 3.3 2.9:
  // Only one of the seq, independent, and auto clauses may appear.
  const auto *Itr =
      llvm::find_if(ExistingClauses,
                    llvm::IsaPred<OpenACCIndependentClause, OpenACCSeqClause>);
  if (Itr != ExistingClauses.end()) {
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_loop_spec_conflict)
        << Clause.getClauseKind() << Clause.getDirectiveKind();
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return nullptr;
  }

  return OpenACCAutoClause::Create(Ctx, Clause.getBeginLoc(),
                                   Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitIndependentClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  // OpenACC 3.3 2.9:
  // Only one of the seq, independent, and auto clauses may appear.
  const auto *Itr = llvm::find_if(
      ExistingClauses, llvm::IsaPred<OpenACCAutoClause, OpenACCSeqClause>);
  if (Itr != ExistingClauses.end()) {
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_loop_spec_conflict)
        << Clause.getClauseKind() << Clause.getDirectiveKind();
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return nullptr;
  }

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
  case OpenACCGangKind::Dim: {
    // OpenACC 3.3 2.9.2: When the parent compute construct is a parallel
    // construct, or an orphaned loop construct, the gang clause behaves as
    // follows. ... The dim argument must be a constant positive integer value
    // 1, 2, or 3.
    if (!E)
      return ExprError();
    ExprResult Res =
        S.ActOnIntExpr(OpenACCDirectiveKind::Invalid, OpenACCClauseKind::Gang,
                       E->getBeginLoc(), E);

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

      S.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
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

OpenACCClause *SemaOpenACCClauseVisitor::VisitVectorClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DiagIfSeqClause(Clause))
    return nullptr;

  // Restrictions only properly implemented on 'loop'/'combined' constructs, and
  // it is the only construct that can do anything with this, so skip/treat as
  // unimplemented for the routine constructs.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

  Expr *IntExpr =
      Clause.getNumIntExprs() != 0 ? Clause.getIntExprs()[0] : nullptr;
  if (IntExpr) {
    if (!isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
      switch (SemaRef.getActiveComputeConstructInfo().Kind) {
      case OpenACCDirectiveKind::Invalid:
      case OpenACCDirectiveKind::Parallel:
        // No restriction on when 'parallel' can contain an argument.
        break;
      case OpenACCDirectiveKind::Serial:
        // GCC disallows this, and there is no real good reason for us to permit
        // it, so disallow until we come up with a use case that makes sense.
        DiagIntArgInvalid(SemaRef, IntExpr, "length", OpenACCClauseKind::Vector,
                          Clause.getDirectiveKind(),
                          SemaRef.getActiveComputeConstructInfo().Kind);
        IntExpr = nullptr;
        break;
      case OpenACCDirectiveKind::Kernels: {
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
                       diag::note_acc_previous_clause_here);

          IntExpr = nullptr;
        }
        break;
      }
      default:
        llvm_unreachable("Non compute construct in active compute construct");
      }
    } else {
      if (Clause.getDirectiveKind() == OpenACCDirectiveKind::SerialLoop) {
        DiagIntArgInvalid(SemaRef, IntExpr, "length", OpenACCClauseKind::Vector,
                          Clause.getDirectiveKind(),
                          SemaRef.getActiveComputeConstructInfo().Kind);
        IntExpr = nullptr;
      } else if (Clause.getDirectiveKind() ==
                 OpenACCDirectiveKind::KernelsLoop) {
        const auto *Itr = llvm::find_if(
            ExistingClauses, llvm::IsaPred<OpenACCVectorLengthClause>);
        if (Itr != ExistingClauses.end()) {
          SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
              << "length" << OpenACCClauseKind::Vector
              << Clause.getDirectiveKind()
              << HasAssocKind(Clause.getDirectiveKind(),
                              SemaRef.getActiveComputeConstructInfo().Kind)
              << SemaRef.getActiveComputeConstructInfo().Kind
              << OpenACCClauseKind::VectorLength;
          SemaRef.Diag((*Itr)->getBeginLoc(),
                       diag::note_acc_previous_clause_here);

          IntExpr = nullptr;
        }
      }
    }
  }

  if (!isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
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
                   diag::note_acc_previous_clause_here);
      return nullptr;
    }
  }

  return OpenACCVectorClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), IntExpr,
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitWorkerClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DiagIfSeqClause(Clause))
    return nullptr;

  // Restrictions only properly implemented on 'loop'/'combined' constructs, and
  // it is the only construct that can do anything with this, so skip/treat as
  // unimplemented for the routine constructs.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

  Expr *IntExpr =
      Clause.getNumIntExprs() != 0 ? Clause.getIntExprs()[0] : nullptr;

  if (IntExpr) {
    if (!isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
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
                       diag::note_acc_previous_clause_here);

          IntExpr = nullptr;
        }
        break;
      }
      default:
        llvm_unreachable("Non compute construct in active compute construct");
      }
    } else {
      if (Clause.getDirectiveKind() == OpenACCDirectiveKind::ParallelLoop ||
          Clause.getDirectiveKind() == OpenACCDirectiveKind::SerialLoop) {
        DiagIntArgInvalid(SemaRef, IntExpr, OpenACCGangKind::Num,
                          OpenACCClauseKind::Worker, Clause.getDirectiveKind(),
                          SemaRef.getActiveComputeConstructInfo().Kind);
        IntExpr = nullptr;
      } else {
        assert(Clause.getDirectiveKind() == OpenACCDirectiveKind::KernelsLoop &&
               "Unknown combined directive kind?");
        const auto *Itr = llvm::find_if(ExistingClauses,
                                        llvm::IsaPred<OpenACCNumWorkersClause>);
        if (Itr != ExistingClauses.end()) {
          SemaRef.Diag(IntExpr->getBeginLoc(), diag::err_acc_num_arg_conflict)
              << "num" << OpenACCClauseKind::Worker << Clause.getDirectiveKind()
              << HasAssocKind(Clause.getDirectiveKind(),
                              SemaRef.getActiveComputeConstructInfo().Kind)
              << SemaRef.getActiveComputeConstructInfo().Kind
              << OpenACCClauseKind::NumWorkers;
          SemaRef.Diag((*Itr)->getBeginLoc(),
                       diag::note_acc_previous_clause_here);

          IntExpr = nullptr;
        }
      }
    }
  }

  if (!isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
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
                   diag::note_acc_previous_clause_here);
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
                   diag::note_acc_previous_clause_here);
      return nullptr;
    }
  }

  return OpenACCWorkerClause::Create(Ctx, Clause.getBeginLoc(),
                                     Clause.getLParenLoc(), IntExpr,
                                     Clause.getEndLoc());
}

OpenACCClause *SemaOpenACCClauseVisitor::VisitGangClause(
    SemaOpenACC::OpenACCParsedClause &Clause) {
  if (DiagIfSeqClause(Clause))
    return nullptr;

  // Restrictions only properly implemented on 'loop' constructs, and it is
  // the only construct that can do anything with this, so skip/treat as
  // unimplemented for the combined constructs.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

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
                     diag::note_acc_previous_clause_here);
        SemaRef.Diag((*NumGangsClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here);
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

  if (!isOpenACCCombinedDirectiveKind(Clause.getDirectiveKind())) {
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
                   diag::note_acc_previous_clause_here);
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
                   diag::note_acc_previous_clause_here);
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
                   diag::note_acc_previous_clause_here);
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
  // Restrictions only properly implemented on 'loop' constructs and combined ,
  // and it is the only construct that can do anything with this, so skip/treat
  // as unimplemented for the routine constructs.
  if (!isDirectiveKindImplemented(Clause.getDirectiveKind()))
    return isNotImplemented();

  // OpenACC 3.3 2.9:
  // Only one of the seq, independent, and auto clauses may appear.
  const auto *Itr =
      llvm::find_if(ExistingClauses,
                    llvm::IsaPred<OpenACCAutoClause, OpenACCIndependentClause>);
  if (Itr != ExistingClauses.end()) {
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_loop_spec_conflict)
        << Clause.getClauseKind() << Clause.getDirectiveKind();
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return nullptr;
  }

  // OpenACC 3.3 2.9:
  // A 'gang', 'worker', or 'vector' clause may not appear if a 'seq' clause
  // appears.
  Itr = llvm::find_if(ExistingClauses,
                      llvm::IsaPred<OpenACCGangClause, OpenACCWorkerClause,
                                    OpenACCVectorClause>);

  if (Itr != ExistingClauses.end()) {
    SemaRef.Diag(Clause.getBeginLoc(), diag::err_acc_clause_cannot_combine)
        << Clause.getClauseKind() << (*Itr)->getClauseKind()
        << Clause.getDirectiveKind();
    SemaRef.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return nullptr;
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
                     diag::note_acc_previous_clause_here);
        SemaRef.Diag((*NumGangsClauseItr)->getBeginLoc(),
                     diag::note_acc_previous_clause_here);
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
        SemaRef.Diag(NGC->getBeginLoc(), diag::note_acc_previous_clause_here);
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
  // Duplicates here are not really sensible.  We could possible permit
  // multiples if they all had the same value, but there isn't really a good
  // reason to do so. Also, this simplifies the suppression of duplicates, in
  // that we know if we 'find' one after instantiation, that it is the same
  // clause, which simplifies instantiation/checking/etc.
  if (checkAlreadyHasClauseOfKind(SemaRef, ExistingClauses, Clause))
    return nullptr;

  ExprResult LoopCount = SemaRef.CheckCollapseLoopCount(Clause.getLoopCount());

  if (!LoopCount.isUsable())
    return nullptr;

  return OpenACCCollapseClause::Create(Ctx, Clause.getBeginLoc(),
                                       Clause.getLParenLoc(), Clause.isForce(),
                                       LoopCount.get(), Clause.getEndLoc());
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

  // Diagnose that we don't support this clause on this directive.
  if (!doesClauseApplyToDirective(Clause.getDirectiveKind(),
                                  Clause.getClauseKind())) {
    Diag(Clause.getBeginLoc(), diag::err_acc_clause_appertainment)
        << Clause.getDirectiveKind() << Clause.getClauseKind();
    return nullptr;
  }

  if (const auto *DevTypeClause =
          llvm::find_if(ExistingClauses,
                        [&](const OpenACCClause *C) {
                          return isa<OpenACCDeviceTypeClause>(C);
                        });
      DevTypeClause != ExistingClauses.end()) {
    if (checkValidAfterDeviceType(
            *this, *cast<OpenACCDeviceTypeClause>(*DevTypeClause), Clause))
      return nullptr;
  }

  SemaOpenACCClauseVisitor Visitor{*this, ExistingClauses};
  OpenACCClause *Result = Visitor.Visit(Clause);
  assert((!Result || Result->getClauseKind() == Clause.getClauseKind()) &&
         "Created wrong clause?");

  if (Visitor.diagNotImplemented())
    Diag(Clause.getBeginLoc(), diag::warn_acc_clause_unimplemented)
        << Clause.getClauseKind();

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
          Diag(OldVarExpr->getExprLoc(), diag::note_acc_previous_clause_here);
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
  default:
    // TODO: OpenACC: when we implement this on 'routine', we'll have to
    // implement its checking here.
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
  // OpenACC 3.3 2.9.11: A reduction clause may not appear on a loop directive
  // that has a gang clause with a dim: argument whose value is greater than 1.

  const auto *ReductionItr =
      llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCReductionClause>);

  if (ReductionItr != ExistingClauses.end()) {
    const auto GangZip = llvm::zip_equal(GangKinds, IntExprs);
    const auto GangItr = llvm::find_if(GangZip, [](const auto &Tuple) {
      return std::get<0>(Tuple) == OpenACCGangKind::Dim;
    });

    if (GangItr != GangZip.end()) {
      const Expr *DimExpr = std::get<1>(*GangItr);

      assert(
          (DimExpr->isInstantiationDependent() || isa<ConstantExpr>(DimExpr)) &&
          "Improperly formed gang argument");
      if (const auto *DimVal = dyn_cast<ConstantExpr>(DimExpr);
          DimVal && DimVal->getResultAsAPSInt() > 1) {
        Diag(DimVal->getBeginLoc(), diag::err_acc_gang_reduction_conflict)
            << /*gang/reduction=*/0 << DirKind;
        Diag((*ReductionItr)->getBeginLoc(),
             diag::note_acc_previous_clause_here);
        return nullptr;
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
          Diag(GangClause->getBeginLoc(), diag::note_acc_previous_clause_here);
          return nullptr;
        }
      }
    }
  }

  auto *Ret = OpenACCReductionClause::Create(
      getASTContext(), BeginLoc, LParenLoc, ReductionOp, Vars, EndLoc);
  return Ret;
}
