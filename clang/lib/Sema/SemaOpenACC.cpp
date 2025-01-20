//===--- SemaOpenACC.cpp - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaOpenACC.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using namespace clang;

namespace {
bool diagnoseConstructAppertainment(SemaOpenACC &S, OpenACCDirectiveKind K,
                                    SourceLocation StartLoc, bool IsStmt) {
  switch (K) {
  default:
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, both invalid and unimplemented don't really need to
    // do anything.
    break;
  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop:
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
  case OpenACCDirectiveKind::Loop:
  case OpenACCDirectiveKind::Data:
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
  case OpenACCDirectiveKind::HostData:
  case OpenACCDirectiveKind::Wait:
    if (!IsStmt)
      return S.Diag(StartLoc, diag::err_acc_construct_appertainment) << K;
    break;
  }
  return false;
}

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
  return isOpenACCComputeDirectiveKind(DK) ||
         isOpenACCCombinedDirectiveKind(DK) || isOpenACCDataDirectiveKind(DK) ||
         DK == OpenACCDirectiveKind::Loop || DK == OpenACCDirectiveKind::Wait ||
         DK == OpenACCDirectiveKind::Init ||
         DK == OpenACCDirectiveKind::Shutdown ||
         DK == OpenACCDirectiveKind::Set;
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
    return ExprError();
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

void CollectActiveReductionClauses(
    llvm::SmallVector<OpenACCReductionClause *> &ActiveClauses,
    ArrayRef<OpenACCClause *> CurClauses) {
  for (auto *CurClause : CurClauses) {
    if (auto *RedClause = dyn_cast<OpenACCReductionClause>(CurClause);
        RedClause && !RedClause->getVarList().empty())
      ActiveClauses.push_back(RedClause);
  }
}

// Depth needs to be preserved for all associated statements that aren't
// supposed to modify the compute/combined/loop construct information.
bool PreserveLoopRAIIDepthInAssociatedStmtRAII(OpenACCDirectiveKind DK) {
  switch (DK) {
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::Kernels:
  case OpenACCDirectiveKind::KernelsLoop:
  case OpenACCDirectiveKind::Loop:
    return false;
  case OpenACCDirectiveKind::Data:
  case OpenACCDirectiveKind::HostData:
    return true;
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
  case OpenACCDirectiveKind::Wait:
  case OpenACCDirectiveKind::Init:
  case OpenACCDirectiveKind::Shutdown:
  case OpenACCDirectiveKind::Set:
  case OpenACCDirectiveKind::Update:
    llvm_unreachable("Doesn't have an associated stmt");
  default:
  case OpenACCDirectiveKind::Invalid:
    llvm_unreachable("Unhandled directive kind?");
  }
  llvm_unreachable("Unhandled directive kind?");
}

} // namespace

SemaOpenACC::SemaOpenACC(Sema &S) : SemaBase(S) {}

SemaOpenACC::AssociatedStmtRAII::AssociatedStmtRAII(
    SemaOpenACC &S, OpenACCDirectiveKind DK, SourceLocation DirLoc,
    ArrayRef<const OpenACCClause *> UnInstClauses,
    ArrayRef<OpenACCClause *> Clauses)
    : SemaRef(S), OldActiveComputeConstructInfo(S.ActiveComputeConstructInfo),
      DirKind(DK), OldLoopGangClauseOnKernel(S.LoopGangClauseOnKernel),
      OldLoopWorkerClauseLoc(S.LoopWorkerClauseLoc),
      OldLoopVectorClauseLoc(S.LoopVectorClauseLoc),
      OldLoopWithoutSeqInfo(S.LoopWithoutSeqInfo),
      ActiveReductionClauses(S.ActiveReductionClauses),
      LoopRAII(SemaRef, PreserveLoopRAIIDepthInAssociatedStmtRAII(DirKind)) {

  // Compute constructs end up taking their 'loop'.
  if (DirKind == OpenACCDirectiveKind::Parallel ||
      DirKind == OpenACCDirectiveKind::Serial ||
      DirKind == OpenACCDirectiveKind::Kernels) {
    CollectActiveReductionClauses(S.ActiveReductionClauses, Clauses);
    SemaRef.ActiveComputeConstructInfo.Kind = DirKind;
    SemaRef.ActiveComputeConstructInfo.Clauses = Clauses;

    // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
    // construct, the gang clause behaves as follows. ... The region of a loop
    // with a gang clause may not contain another loop with a gang clause unless
    // within a nested compute region.
    //
    // Implement the 'unless within a nested compute region' part.
    SemaRef.LoopGangClauseOnKernel = {};
    SemaRef.LoopWorkerClauseLoc = {};
    SemaRef.LoopVectorClauseLoc = {};
    SemaRef.LoopWithoutSeqInfo = {};
  } else if (DirKind == OpenACCDirectiveKind::ParallelLoop ||
             DirKind == OpenACCDirectiveKind::SerialLoop ||
             DirKind == OpenACCDirectiveKind::KernelsLoop) {
    SemaRef.ActiveComputeConstructInfo.Kind = DirKind;
    SemaRef.ActiveComputeConstructInfo.Clauses = Clauses;

    CollectActiveReductionClauses(S.ActiveReductionClauses, Clauses);
    SetCollapseInfoBeforeAssociatedStmt(UnInstClauses, Clauses);
    SetTileInfoBeforeAssociatedStmt(UnInstClauses, Clauses);

    SemaRef.LoopGangClauseOnKernel = {};
    SemaRef.LoopWorkerClauseLoc = {};
    SemaRef.LoopVectorClauseLoc = {};

    // Set the active 'loop' location if there isn't a 'seq' on it, so we can
    // diagnose the for loops.
    SemaRef.LoopWithoutSeqInfo = {};
    if (Clauses.end() ==
        llvm::find_if(Clauses, llvm::IsaPred<OpenACCSeqClause>))
      SemaRef.LoopWithoutSeqInfo = {DirKind, DirLoc};

    // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
    // construct, the gang clause behaves as follows. ... The region of a loop
    // with a gang clause may not contain another loop with a gang clause unless
    // within a nested compute region.
    //
    // We don't bother doing this when this is a template instantiation, as
    // there is no reason to do these checks: the existance of a
    // gang/kernels/etc cannot be dependent.
    if (DirKind == OpenACCDirectiveKind::KernelsLoop && UnInstClauses.empty()) {
      // This handles the 'outer loop' part of this.
      auto *Itr = llvm::find_if(Clauses, llvm::IsaPred<OpenACCGangClause>);
      if (Itr != Clauses.end())
        SemaRef.LoopGangClauseOnKernel = {(*Itr)->getBeginLoc(), DirKind};
    }

    if (UnInstClauses.empty()) {
      auto *Itr = llvm::find_if(Clauses, llvm::IsaPred<OpenACCWorkerClause>);
      if (Itr != Clauses.end())
        SemaRef.LoopWorkerClauseLoc = (*Itr)->getBeginLoc();

      auto *Itr2 = llvm::find_if(Clauses, llvm::IsaPred<OpenACCVectorClause>);
      if (Itr2 != Clauses.end())
        SemaRef.LoopVectorClauseLoc = (*Itr2)->getBeginLoc();
    }
  } else if (DirKind == OpenACCDirectiveKind::Loop) {
    CollectActiveReductionClauses(S.ActiveReductionClauses, Clauses);
    SetCollapseInfoBeforeAssociatedStmt(UnInstClauses, Clauses);
    SetTileInfoBeforeAssociatedStmt(UnInstClauses, Clauses);

    // Set the active 'loop' location if there isn't a 'seq' on it, so we can
    // diagnose the for loops.
    SemaRef.LoopWithoutSeqInfo = {};
    if (Clauses.end() ==
        llvm::find_if(Clauses, llvm::IsaPred<OpenACCSeqClause>))
      SemaRef.LoopWithoutSeqInfo = {DirKind, DirLoc};

    // OpenACC 3.3 2.9.2: When the parent compute construct is a kernels
    // construct, the gang clause behaves as follows. ... The region of a loop
    // with a gang clause may not contain another loop with a gang clause unless
    // within a nested compute region.
    //
    // We don't bother doing this when this is a template instantiation, as
    // there is no reason to do these checks: the existance of a
    // gang/kernels/etc cannot be dependent.
    if (SemaRef.getActiveComputeConstructInfo().Kind ==
            OpenACCDirectiveKind::Kernels &&
        UnInstClauses.empty()) {
      // This handles the 'outer loop' part of this.
      auto *Itr = llvm::find_if(Clauses, llvm::IsaPred<OpenACCGangClause>);
      if (Itr != Clauses.end())
        SemaRef.LoopGangClauseOnKernel = {(*Itr)->getBeginLoc(),
                                          OpenACCDirectiveKind::Kernels};
    }

    if (UnInstClauses.empty()) {
      auto *Itr = llvm::find_if(Clauses, llvm::IsaPred<OpenACCWorkerClause>);
      if (Itr != Clauses.end())
        SemaRef.LoopWorkerClauseLoc = (*Itr)->getBeginLoc();

      auto *Itr2 = llvm::find_if(Clauses, llvm::IsaPred<OpenACCVectorClause>);
      if (Itr2 != Clauses.end())
        SemaRef.LoopVectorClauseLoc = (*Itr2)->getBeginLoc();
    }
  }
}

void SemaOpenACC::AssociatedStmtRAII::SetCollapseInfoBeforeAssociatedStmt(
    ArrayRef<const OpenACCClause *> UnInstClauses,
    ArrayRef<OpenACCClause *> Clauses) {

  // Reset this checking for loops that aren't covered in a RAII object.
  SemaRef.LoopInfo.CurLevelHasLoopAlready = false;
  SemaRef.CollapseInfo.CollapseDepthSatisfied = true;
  SemaRef.TileInfo.TileDepthSatisfied = true;

  // We make sure to take an optional list of uninstantiated clauses, so that
  // we can check to make sure we don't 'double diagnose' in the event that
  // the value of 'N' was not dependent in a template. We also ensure during
  // Sema that there is only 1 collapse on each construct, so we can count on
  // the fact that if both find a 'collapse', that they are the same one.
  auto *CollapseClauseItr =
      llvm::find_if(Clauses, llvm::IsaPred<OpenACCCollapseClause>);
  auto *UnInstCollapseClauseItr =
      llvm::find_if(UnInstClauses, llvm::IsaPred<OpenACCCollapseClause>);

  if (Clauses.end() == CollapseClauseItr)
    return;

  OpenACCCollapseClause *CollapseClause =
      cast<OpenACCCollapseClause>(*CollapseClauseItr);

  SemaRef.CollapseInfo.ActiveCollapse = CollapseClause;
  Expr *LoopCount = CollapseClause->getLoopCount();

  // If the loop count is still instantiation dependent, setting the depth
  // counter isn't necessary, so return here.
  if (!LoopCount || LoopCount->isInstantiationDependent())
    return;

  // Suppress diagnostics if we've done a 'transform' where the previous version
  // wasn't dependent, meaning we already diagnosed it.
  if (UnInstCollapseClauseItr != UnInstClauses.end() &&
      !cast<OpenACCCollapseClause>(*UnInstCollapseClauseItr)
           ->getLoopCount()
           ->isInstantiationDependent())
    return;

  SemaRef.CollapseInfo.CollapseDepthSatisfied = false;
  SemaRef.CollapseInfo.CurCollapseCount =
      cast<ConstantExpr>(LoopCount)->getResultAsAPSInt();
  SemaRef.CollapseInfo.DirectiveKind = DirKind;
}

void SemaOpenACC::AssociatedStmtRAII::SetTileInfoBeforeAssociatedStmt(
    ArrayRef<const OpenACCClause *> UnInstClauses,
    ArrayRef<OpenACCClause *> Clauses) {
  // We don't diagnose if this is during instantiation, since the only thing we
  // care about is the number of arguments, which we can figure out without
  // instantiation, so we don't want to double-diagnose.
  if (UnInstClauses.size() > 0)
    return;
  auto *TileClauseItr =
      llvm::find_if(Clauses, llvm::IsaPred<OpenACCTileClause>);

  if (Clauses.end() == TileClauseItr)
    return;

  OpenACCTileClause *TileClause = cast<OpenACCTileClause>(*TileClauseItr);
  SemaRef.TileInfo.ActiveTile = TileClause;
  SemaRef.TileInfo.TileDepthSatisfied = false;
  SemaRef.TileInfo.CurTileCount = TileClause->getSizeExprs().size();
  SemaRef.TileInfo.DirectiveKind = DirKind;
}

SemaOpenACC::AssociatedStmtRAII::~AssociatedStmtRAII() {
  if (DirKind == OpenACCDirectiveKind::Parallel ||
      DirKind == OpenACCDirectiveKind::Serial ||
      DirKind == OpenACCDirectiveKind::Kernels ||
      DirKind == OpenACCDirectiveKind::Loop ||
      DirKind == OpenACCDirectiveKind::ParallelLoop ||
      DirKind == OpenACCDirectiveKind::SerialLoop ||
      DirKind == OpenACCDirectiveKind::KernelsLoop) {
    SemaRef.ActiveComputeConstructInfo = OldActiveComputeConstructInfo;
    SemaRef.LoopGangClauseOnKernel = OldLoopGangClauseOnKernel;
    SemaRef.LoopWorkerClauseLoc = OldLoopWorkerClauseLoc;
    SemaRef.LoopVectorClauseLoc = OldLoopVectorClauseLoc;
    SemaRef.LoopWithoutSeqInfo = OldLoopWithoutSeqInfo;
    SemaRef.ActiveReductionClauses.swap(ActiveReductionClauses);
  } else if (DirKind == OpenACCDirectiveKind::Data ||
             DirKind == OpenACCDirectiveKind::HostData) {
    // Intentionally doesn't reset the Loop, Compute Construct, or reduction
    // effects.
  }
}

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

namespace {
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

void SemaOpenACC::ActOnConstruct(OpenACCDirectiveKind K,
                                 SourceLocation DirLoc) {
  // Start an evaluation context to parse the clause arguments on.
  SemaRef.PushExpressionEvaluationContext(
      Sema::ExpressionEvaluationContext::PotentiallyEvaluated);

  switch (K) {
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, an invalid kind has nothing we can check here.  We
    // want to continue parsing clauses as far as we can, so we will just
    // ensure that we can still work and don't check any construct-specific
    // rules anywhere.
    break;
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop:
  case OpenACCDirectiveKind::Loop:
  case OpenACCDirectiveKind::Data:
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
  case OpenACCDirectiveKind::HostData:
  case OpenACCDirectiveKind::Init:
  case OpenACCDirectiveKind::Shutdown:
  case OpenACCDirectiveKind::Set:
  case OpenACCDirectiveKind::Update:
    // Nothing to do here, there is no real legalization that needs to happen
    // here as these constructs do not take any arguments.
    break;
  case OpenACCDirectiveKind::Wait:
    // Nothing really to do here, the arguments to the 'wait' should have
    // already been handled by the time we get here.
    break;
  default:
    Diag(DirLoc, diag::warn_acc_construct_unimplemented) << K;
    break;
  }
}

ExprResult SemaOpenACC::ActOnIntExpr(OpenACCDirectiveKind DK,
                                     OpenACCClauseKind CK, SourceLocation Loc,
                                     Expr *IntExpr) {

  assert(((DK != OpenACCDirectiveKind::Invalid &&
           CK == OpenACCClauseKind::Invalid) ||
          (DK == OpenACCDirectiveKind::Invalid &&
           CK != OpenACCClauseKind::Invalid) ||
          (DK == OpenACCDirectiveKind::Invalid &&
           CK == OpenACCClauseKind::Invalid)) &&
         "Only one of directive or clause kind should be provided");

  class IntExprConverter : public Sema::ICEConvertDiagnoser {
    OpenACCDirectiveKind DirectiveKind;
    OpenACCClauseKind ClauseKind;
    Expr *IntExpr;

    // gets the index into the diagnostics so we can use this for clauses,
    // directives, and sub array.s
    unsigned getDiagKind() const {
      if (ClauseKind != OpenACCClauseKind::Invalid)
        return 0;
      if (DirectiveKind != OpenACCDirectiveKind::Invalid)
        return 1;
      return 2;
    }

  public:
    IntExprConverter(OpenACCDirectiveKind DK, OpenACCClauseKind CK,
                     Expr *IntExpr)
        : ICEConvertDiagnoser(/*AllowScopedEnumerations=*/false,
                              /*Suppress=*/false,
                              /*SuppressConversion=*/true),
          DirectiveKind(DK), ClauseKind(CK), IntExpr(IntExpr) {}

    bool match(QualType T) override {
      // OpenACC spec just calls this 'integer expression' as having an
      // 'integer type', so fall back on C99's 'integer type'.
      return T->isIntegerType();
    }
    SemaBase::SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                   QualType T) override {
      return S.Diag(Loc, diag::err_acc_int_expr_requires_integer)
             << getDiagKind() << ClauseKind << DirectiveKind << T;
    }

    SemaBase::SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) override {
      return S.Diag(Loc, diag::err_acc_int_expr_incomplete_class_type)
             << T << IntExpr->getSourceRange();
    }

    SemaBase::SemaDiagnosticBuilder
    diagnoseExplicitConv(Sema &S, SourceLocation Loc, QualType T,
                         QualType ConvTy) override {
      return S.Diag(Loc, diag::err_acc_int_expr_explicit_conversion)
             << T << ConvTy;
    }

    SemaBase::SemaDiagnosticBuilder noteExplicitConv(Sema &S,
                                                     CXXConversionDecl *Conv,
                                                     QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_acc_int_expr_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    SemaBase::SemaDiagnosticBuilder
    diagnoseAmbiguous(Sema &S, SourceLocation Loc, QualType T) override {
      return S.Diag(Loc, diag::err_acc_int_expr_multiple_conversions) << T;
    }

    SemaBase::SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_acc_int_expr_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    SemaBase::SemaDiagnosticBuilder
    diagnoseConversion(Sema &S, SourceLocation Loc, QualType T,
                       QualType ConvTy) override {
      llvm_unreachable("conversion functions are permitted");
    }
  } IntExprDiagnoser(DK, CK, IntExpr);

  if (!IntExpr)
    return ExprError();

  ExprResult IntExprResult = SemaRef.PerformContextualImplicitConversion(
      Loc, IntExpr, IntExprDiagnoser);
  if (IntExprResult.isInvalid())
    return ExprError();

  IntExpr = IntExprResult.get();
  if (!IntExpr->isTypeDependent() && !IntExpr->getType()->isIntegerType())
    return ExprError();

  // TODO OpenACC: Do we want to perform usual unary conversions here? When
  // doing codegen we might find that is necessary, but skip it for now.
  return IntExpr;
}

bool SemaOpenACC::CheckVarIsPointerType(OpenACCClauseKind ClauseKind,
                                        Expr *VarExpr) {
  // We already know that VarExpr is a proper reference to a variable, so we
  // should be able to just take the type of the expression to get the type of
  // the referenced variable.

  // We've already seen an error, don't diagnose anything else.
  if (!VarExpr || VarExpr->containsErrors())
    return false;

  if (isa<ArraySectionExpr>(VarExpr->IgnoreParenImpCasts()) ||
      VarExpr->hasPlaceholderType(BuiltinType::ArraySection)) {
    Diag(VarExpr->getExprLoc(), diag::err_array_section_use) << /*OpenACC=*/0;
    Diag(VarExpr->getExprLoc(), diag::note_acc_expected_pointer_var);
    return true;
  }

  QualType Ty = VarExpr->getType();
  Ty = Ty.getNonReferenceType().getUnqualifiedType();

  // Nothing we can do if this is a dependent type.
  if (Ty->isDependentType())
    return false;

  if (!Ty->isPointerType())
    return Diag(VarExpr->getExprLoc(), diag::err_acc_var_not_pointer_type)
           << ClauseKind << Ty;
  return false;
}

ExprResult SemaOpenACC::ActOnVar(OpenACCClauseKind CK, Expr *VarExpr) {
  Expr *CurVarExpr = VarExpr->IgnoreParenImpCasts();

  // 'use_device' doesn't allow array subscript or array sections.
  // OpenACC3.3 2.8:
  // A 'var' in a 'use_device' clause must be the name of a variable or array.
  if (CK == OpenACCClauseKind::UseDevice &&
      isa<ArraySectionExpr, ArraySubscriptExpr>(CurVarExpr)) {
    Diag(VarExpr->getExprLoc(), diag::err_acc_not_a_var_ref_use_device);
    return ExprError();
  }

  // Sub-arrays/subscript-exprs are fine as long as the base is a
  // VarExpr/MemberExpr. So strip all of those off.
  while (isa<ArraySectionExpr, ArraySubscriptExpr>(CurVarExpr)) {
    if (auto *SubScrpt = dyn_cast<ArraySubscriptExpr>(CurVarExpr))
      CurVarExpr = SubScrpt->getBase()->IgnoreParenImpCasts();
    else
      CurVarExpr =
          cast<ArraySectionExpr>(CurVarExpr)->getBase()->IgnoreParenImpCasts();
  }

  // References to a VarDecl are fine.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(CurVarExpr)) {
    if (isa<VarDecl, NonTypeTemplateParmDecl>(
            DRE->getFoundDecl()->getCanonicalDecl()))
      return VarExpr;
  }

  // If CK is a Reduction, this special cases for OpenACC3.3 2.5.15: "A var in a
  // reduction clause must be a scalar variable name, an aggregate variable
  // name, an array element, or a subarray.
  // If CK is a 'use_device', this also isn't valid, as it isn' the name of a
  // variable or array.
  // A MemberExpr that references a Field is valid for other clauses.
  if (CK != OpenACCClauseKind::Reduction &&
      CK != OpenACCClauseKind::UseDevice) {
    if (const auto *ME = dyn_cast<MemberExpr>(CurVarExpr)) {
      if (isa<FieldDecl>(ME->getMemberDecl()->getCanonicalDecl()))
        return VarExpr;
    }
  }

  // Referring to 'this' is ok for the most part, but for 'use_device' doesn't
  // fall into 'variable or array name'
  if (CK != OpenACCClauseKind::UseDevice && isa<CXXThisExpr>(CurVarExpr))
    return VarExpr;

  // Nothing really we can do here, as these are dependent.  So just return they
  // are valid.
  if (isa<DependentScopeDeclRefExpr>(CurVarExpr) ||
      (CK != OpenACCClauseKind::Reduction &&
       isa<CXXDependentScopeMemberExpr>(CurVarExpr)))
    return VarExpr;

  // There isn't really anything we can do in the case of a recovery expr, so
  // skip the diagnostic rather than produce a confusing diagnostic.
  if (isa<RecoveryExpr>(CurVarExpr))
    return ExprError();

  if (CK == OpenACCClauseKind::UseDevice)
    Diag(VarExpr->getExprLoc(), diag::err_acc_not_a_var_ref_use_device);
  else
    Diag(VarExpr->getExprLoc(), diag::err_acc_not_a_var_ref)
        << (CK != OpenACCClauseKind::Reduction);
  return ExprError();
}

ExprResult SemaOpenACC::ActOnArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                              Expr *LowerBound,
                                              SourceLocation ColonLoc,
                                              Expr *Length,
                                              SourceLocation RBLoc) {
  ASTContext &Context = getASTContext();

  // Handle placeholders.
  if (Base->hasPlaceholderType() &&
      !Base->hasPlaceholderType(BuiltinType::ArraySection)) {
    ExprResult Result = SemaRef.CheckPlaceholderExpr(Base);
    if (Result.isInvalid())
      return ExprError();
    Base = Result.get();
  }
  if (LowerBound && LowerBound->getType()->isNonOverloadPlaceholderType()) {
    ExprResult Result = SemaRef.CheckPlaceholderExpr(LowerBound);
    if (Result.isInvalid())
      return ExprError();
    Result = SemaRef.DefaultLvalueConversion(Result.get());
    if (Result.isInvalid())
      return ExprError();
    LowerBound = Result.get();
  }
  if (Length && Length->getType()->isNonOverloadPlaceholderType()) {
    ExprResult Result = SemaRef.CheckPlaceholderExpr(Length);
    if (Result.isInvalid())
      return ExprError();
    Result = SemaRef.DefaultLvalueConversion(Result.get());
    if (Result.isInvalid())
      return ExprError();
    Length = Result.get();
  }

  // Check the 'base' value, it must be an array or pointer type, and not to/of
  // a function type.
  QualType OriginalBaseTy = ArraySectionExpr::getBaseOriginalType(Base);
  QualType ResultTy;
  if (!Base->isTypeDependent()) {
    if (OriginalBaseTy->isAnyPointerType()) {
      ResultTy = OriginalBaseTy->getPointeeType();
    } else if (OriginalBaseTy->isArrayType()) {
      ResultTy = OriginalBaseTy->getAsArrayTypeUnsafe()->getElementType();
    } else {
      return ExprError(
          Diag(Base->getExprLoc(), diag::err_acc_typecheck_subarray_value)
          << Base->getSourceRange());
    }

    if (ResultTy->isFunctionType()) {
      Diag(Base->getExprLoc(), diag::err_acc_subarray_function_type)
          << ResultTy << Base->getSourceRange();
      return ExprError();
    }

    if (SemaRef.RequireCompleteType(Base->getExprLoc(), ResultTy,
                                    diag::err_acc_subarray_incomplete_type,
                                    Base))
      return ExprError();

    if (!Base->hasPlaceholderType(BuiltinType::ArraySection)) {
      ExprResult Result = SemaRef.DefaultFunctionArrayLvalueConversion(Base);
      if (Result.isInvalid())
        return ExprError();
      Base = Result.get();
    }
  }

  auto GetRecovery = [&](Expr *E, QualType Ty) {
    ExprResult Recovery =
        SemaRef.CreateRecoveryExpr(E->getBeginLoc(), E->getEndLoc(), E, Ty);
    return Recovery.isUsable() ? Recovery.get() : nullptr;
  };

  // Ensure both of the expressions are int-exprs.
  if (LowerBound && !LowerBound->isTypeDependent()) {
    ExprResult LBRes =
        ActOnIntExpr(OpenACCDirectiveKind::Invalid, OpenACCClauseKind::Invalid,
                     LowerBound->getExprLoc(), LowerBound);

    if (LBRes.isUsable())
      LBRes = SemaRef.DefaultLvalueConversion(LBRes.get());
    LowerBound =
        LBRes.isUsable() ? LBRes.get() : GetRecovery(LowerBound, Context.IntTy);
  }

  if (Length && !Length->isTypeDependent()) {
    ExprResult LenRes =
        ActOnIntExpr(OpenACCDirectiveKind::Invalid, OpenACCClauseKind::Invalid,
                     Length->getExprLoc(), Length);

    if (LenRes.isUsable())
      LenRes = SemaRef.DefaultLvalueConversion(LenRes.get());
    Length =
        LenRes.isUsable() ? LenRes.get() : GetRecovery(Length, Context.IntTy);
  }

  // Length is required if the base type is not an array of known bounds.
  if (!Length && (OriginalBaseTy.isNull() ||
                  (!OriginalBaseTy->isDependentType() &&
                   !OriginalBaseTy->isConstantArrayType() &&
                   !OriginalBaseTy->isDependentSizedArrayType()))) {
    bool IsArray = !OriginalBaseTy.isNull() && OriginalBaseTy->isArrayType();
    Diag(ColonLoc, diag::err_acc_subarray_no_length) << IsArray;
    // Fill in a dummy 'length' so that when we instantiate this we don't
    // double-diagnose here.
    ExprResult Recovery = SemaRef.CreateRecoveryExpr(
        ColonLoc, SourceLocation(), ArrayRef<Expr *>(), Context.IntTy);
    Length = Recovery.isUsable() ? Recovery.get() : nullptr;
  }

  // Check the values of each of the arguments, they cannot be negative(we
  // assume), and if the array bound is known, must be within range. As we do
  // so, do our best to continue with evaluation, we can set the
  // value/expression to nullptr/nullopt if they are invalid, and treat them as
  // not present for the rest of evaluation.

  // We don't have to check for dependence, because the dependent size is
  // represented as a different AST node.
  std::optional<llvm::APSInt> BaseSize;
  if (!OriginalBaseTy.isNull() && OriginalBaseTy->isConstantArrayType()) {
    const auto *ArrayTy = Context.getAsConstantArrayType(OriginalBaseTy);
    BaseSize = ArrayTy->getSize();
  }

  auto GetBoundValue = [&](Expr *E) -> std::optional<llvm::APSInt> {
    if (!E || E->isInstantiationDependent())
      return std::nullopt;

    Expr::EvalResult Res;
    if (!E->EvaluateAsInt(Res, Context))
      return std::nullopt;
    return Res.Val.getInt();
  };

  std::optional<llvm::APSInt> LowerBoundValue = GetBoundValue(LowerBound);
  std::optional<llvm::APSInt> LengthValue = GetBoundValue(Length);

  // Check lower bound for negative or out of range.
  if (LowerBoundValue.has_value()) {
    if (LowerBoundValue->isNegative()) {
      Diag(LowerBound->getExprLoc(), diag::err_acc_subarray_negative)
          << /*LowerBound=*/0 << toString(*LowerBoundValue, /*Radix=*/10);
      LowerBoundValue.reset();
      LowerBound = GetRecovery(LowerBound, LowerBound->getType());
    } else if (BaseSize.has_value() &&
               llvm::APSInt::compareValues(*LowerBoundValue, *BaseSize) >= 0) {
      // Lower bound (start index) must be less than the size of the array.
      Diag(LowerBound->getExprLoc(), diag::err_acc_subarray_out_of_range)
          << /*LowerBound=*/0 << toString(*LowerBoundValue, /*Radix=*/10)
          << toString(*BaseSize, /*Radix=*/10);
      LowerBoundValue.reset();
      LowerBound = GetRecovery(LowerBound, LowerBound->getType());
    }
  }

  // Check length for negative or out of range.
  if (LengthValue.has_value()) {
    if (LengthValue->isNegative()) {
      Diag(Length->getExprLoc(), diag::err_acc_subarray_negative)
          << /*Length=*/1 << toString(*LengthValue, /*Radix=*/10);
      LengthValue.reset();
      Length = GetRecovery(Length, Length->getType());
    } else if (BaseSize.has_value() &&
               llvm::APSInt::compareValues(*LengthValue, *BaseSize) > 0) {
      // Length must be lessthan or EQUAL to the size of the array.
      Diag(Length->getExprLoc(), diag::err_acc_subarray_out_of_range)
          << /*Length=*/1 << toString(*LengthValue, /*Radix=*/10)
          << toString(*BaseSize, /*Radix=*/10);
      LengthValue.reset();
      Length = GetRecovery(Length, Length->getType());
    }
  }

  // Adding two APSInts requires matching sign, so extract that here.
  auto AddAPSInt = [](llvm::APSInt LHS, llvm::APSInt RHS) -> llvm::APSInt {
    if (LHS.isSigned() == RHS.isSigned())
      return LHS + RHS;

    unsigned Width = std::max(LHS.getBitWidth(), RHS.getBitWidth()) + 1;
    return llvm::APSInt(LHS.sext(Width) + RHS.sext(Width), /*Signed=*/true);
  };

  // If we know all 3 values, we can diagnose that the total value would be out
  // of range.
  if (BaseSize.has_value() && LowerBoundValue.has_value() &&
      LengthValue.has_value() &&
      llvm::APSInt::compareValues(AddAPSInt(*LowerBoundValue, *LengthValue),
                                  *BaseSize) > 0) {
    Diag(Base->getExprLoc(),
         diag::err_acc_subarray_base_plus_length_out_of_range)
        << toString(*LowerBoundValue, /*Radix=*/10)
        << toString(*LengthValue, /*Radix=*/10)
        << toString(*BaseSize, /*Radix=*/10);

    LowerBoundValue.reset();
    LowerBound = GetRecovery(LowerBound, LowerBound->getType());
    LengthValue.reset();
    Length = GetRecovery(Length, Length->getType());
  }

  // If any part of the expression is dependent, return a dependent sub-array.
  QualType ArrayExprTy = Context.ArraySectionTy;
  if (Base->isTypeDependent() ||
      (LowerBound && LowerBound->isInstantiationDependent()) ||
      (Length && Length->isInstantiationDependent()))
    ArrayExprTy = Context.DependentTy;

  return new (Context)
      ArraySectionExpr(Base, LowerBound, Length, ArrayExprTy, VK_LValue,
                       OK_Ordinary, ColonLoc, RBLoc);
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

void SemaOpenACC::ActOnWhileStmt(SourceLocation WhileLoc) {
  if (!getLangOpts().OpenACC)
    return;

  if (!LoopInfo.TopLevelLoopSeen)
    return;

  if (CollapseInfo.CurCollapseCount && *CollapseInfo.CurCollapseCount > 0) {
    Diag(WhileLoc, diag::err_acc_invalid_in_loop)
        << /*while loop*/ 1 << CollapseInfo.DirectiveKind
        << OpenACCClauseKind::Collapse;
    assert(CollapseInfo.ActiveCollapse && "Collapse count without object?");
    Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
         diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Collapse;

    // Remove the value so that we don't get cascading errors in the body. The
    // caller RAII object will restore this.
    CollapseInfo.CurCollapseCount = std::nullopt;
  }

  if (TileInfo.CurTileCount && *TileInfo.CurTileCount > 0) {
    Diag(WhileLoc, diag::err_acc_invalid_in_loop)
        << /*while loop*/ 1 << TileInfo.DirectiveKind
        << OpenACCClauseKind::Tile;
    assert(TileInfo.ActiveTile && "tile count without object?");
    Diag(TileInfo.ActiveTile->getBeginLoc(), diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Tile;

    // Remove the value so that we don't get cascading errors in the body. The
    // caller RAII object will restore this.
    TileInfo.CurTileCount = std::nullopt;
  }
}

void SemaOpenACC::ActOnDoStmt(SourceLocation DoLoc) {
  if (!getLangOpts().OpenACC)
    return;

  if (!LoopInfo.TopLevelLoopSeen)
    return;

  if (CollapseInfo.CurCollapseCount && *CollapseInfo.CurCollapseCount > 0) {
    Diag(DoLoc, diag::err_acc_invalid_in_loop)
        << /*do loop*/ 2 << CollapseInfo.DirectiveKind
        << OpenACCClauseKind::Collapse;
    assert(CollapseInfo.ActiveCollapse && "Collapse count without object?");
    Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
         diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Collapse;

    // Remove the value so that we don't get cascading errors in the body. The
    // caller RAII object will restore this.
    CollapseInfo.CurCollapseCount = std::nullopt;
  }

  if (TileInfo.CurTileCount && *TileInfo.CurTileCount > 0) {
    Diag(DoLoc, diag::err_acc_invalid_in_loop)
        << /*do loop*/ 2 << TileInfo.DirectiveKind << OpenACCClauseKind::Tile;
    assert(TileInfo.ActiveTile && "tile count without object?");
    Diag(TileInfo.ActiveTile->getBeginLoc(), diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Tile;

    // Remove the value so that we don't get cascading errors in the body. The
    // caller RAII object will restore this.
    TileInfo.CurTileCount = std::nullopt;
  }
}

void SemaOpenACC::ForStmtBeginHelper(SourceLocation ForLoc,
                                     ForStmtBeginChecker &C) {
  assert(getLangOpts().OpenACC && "Check enabled when not OpenACC?");

  // Enable the while/do-while checking.
  LoopInfo.TopLevelLoopSeen = true;

  if (CollapseInfo.CurCollapseCount && *CollapseInfo.CurCollapseCount > 0) {
    C.check();

    // OpenACC 3.3 2.9.1:
    // Each associated loop, except the innermost, must contain exactly one loop
    // or loop nest.
    // This checks for more than 1 loop at the current level, the
    // 'depth'-satisifed checking manages the 'not zero' case.
    if (LoopInfo.CurLevelHasLoopAlready) {
      Diag(ForLoc, diag::err_acc_clause_multiple_loops)
          << CollapseInfo.DirectiveKind << OpenACCClauseKind::Collapse;
      assert(CollapseInfo.ActiveCollapse && "No collapse object?");
      Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
           diag::note_acc_active_clause_here)
          << OpenACCClauseKind::Collapse;
    } else {
      --(*CollapseInfo.CurCollapseCount);

      // Once we've hit zero here, we know we have deep enough 'for' loops to
      // get to the bottom.
      if (*CollapseInfo.CurCollapseCount == 0)
        CollapseInfo.CollapseDepthSatisfied = true;
    }
  }

  if (TileInfo.CurTileCount && *TileInfo.CurTileCount > 0) {
    C.check();

    if (LoopInfo.CurLevelHasLoopAlready) {
      Diag(ForLoc, diag::err_acc_clause_multiple_loops)
          << TileInfo.DirectiveKind << OpenACCClauseKind::Tile;
      assert(TileInfo.ActiveTile && "No tile object?");
      Diag(TileInfo.ActiveTile->getBeginLoc(),
           diag::note_acc_active_clause_here)
          << OpenACCClauseKind::Tile;
    } else {
      --(*TileInfo.CurTileCount);
      // Once we've hit zero here, we know we have deep enough 'for' loops to
      // get to the bottom.
      if (*TileInfo.CurTileCount == 0)
        TileInfo.TileDepthSatisfied = true;
    }
  }

  // Set this to 'false' for the body of this loop, so that the next level
  // checks independently.
  LoopInfo.CurLevelHasLoopAlready = false;
}

namespace {
bool isValidLoopVariableType(QualType LoopVarTy) {
  // Just skip if it is dependent, it could be any of the below.
  if (LoopVarTy->isDependentType())
    return true;

  // The loop variable must be of integer,
  if (LoopVarTy->isIntegerType())
    return true;

  // C/C++ pointer,
  if (LoopVarTy->isPointerType())
    return true;

  // or C++ random-access iterator type.
  if (const auto *RD = LoopVarTy->getAsCXXRecordDecl()) {
    // Note: Only do CXXRecordDecl because RecordDecl can't be a random access
    // iterator type!

    // We could either do a lot of work to see if this matches
    // random-access-iterator, but it seems that just checking that the
    // 'iterator_category' typedef is more than sufficient. If programmers are
    // willing to lie about this, we can let them.

    for (const auto *TD :
         llvm::make_filter_range(RD->decls(), llvm::IsaPred<TypedefNameDecl>)) {
      const auto *TDND = cast<TypedefNameDecl>(TD)->getCanonicalDecl();

      if (TDND->getName() != "iterator_category")
        continue;

      // If there is no type for this decl, return false.
      if (TDND->getUnderlyingType().isNull())
        return false;

      const CXXRecordDecl *ItrCategoryDecl =
          TDND->getUnderlyingType()->getAsCXXRecordDecl();

      // If the category isn't a record decl, it isn't the tag type.
      if (!ItrCategoryDecl)
        return false;

      auto IsRandomAccessIteratorTag = [](const CXXRecordDecl *RD) {
        if (RD->getName() != "random_access_iterator_tag")
          return false;
        // Checks just for std::random_access_iterator_tag.
        return RD->getEnclosingNamespaceContext()->isStdNamespace();
      };

      if (IsRandomAccessIteratorTag(ItrCategoryDecl))
        return true;

      // We can also support types inherited from the
      // random_access_iterator_tag.
      for (CXXBaseSpecifier BS : ItrCategoryDecl->bases()) {

        if (IsRandomAccessIteratorTag(BS.getType()->getAsCXXRecordDecl()))
          return true;
      }

      return false;
    }
  }

  return false;
}

} // namespace

void SemaOpenACC::ForStmtBeginChecker::check() {
  if (SemaRef.LoopWithoutSeqInfo.Kind == OpenACCDirectiveKind::Invalid)
    return;

  if (AlreadyChecked)
    return;
  AlreadyChecked = true;

  // OpenACC3.3 2.1:
  // A loop associated with a loop construct that does not have a seq clause
  // must be written to meet all the following conditions:
  // - The loop variable must be of integer, C/C++ pointer, or C++ random-access
  // iterator type.
  // - The loop variable must monotonically increase or decrease in the
  // direction of its termination condition.
  // - The loop trip count must be computable in constant time when entering the
  // loop construct.
  //
  // For a C++ range-based for loop, the loop variable
  // identified by the above conditions is the internal iterator, such as a
  // pointer, that the compiler generates to iterate the range.  it is not the
  // variable declared by the for loop.

  if (IsRangeFor) {
    // If the range-for is being instantiated and didn't change, don't
    // re-diagnose.
    if (!RangeFor.has_value())
      return;
    // For a range-for, we can assume everything is 'corect' other than the type
    // of the iterator, so check that.
    const DeclStmt *RangeStmt = (*RangeFor)->getBeginStmt();

    // In some dependent contexts, the autogenerated range statement doesn't get
    // included until instantiation, so skip for now.
    if (!RangeStmt)
      return;

    const ValueDecl *InitVar = cast<ValueDecl>(RangeStmt->getSingleDecl());
    QualType VarType = InitVar->getType().getNonReferenceType();
    if (!isValidLoopVariableType(VarType)) {
      SemaRef.Diag(InitVar->getBeginLoc(), diag::err_acc_loop_variable_type)
          << SemaRef.LoopWithoutSeqInfo.Kind << VarType;
      SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc,
                   diag::note_acc_construct_here)
          << SemaRef.LoopWithoutSeqInfo.Kind;
    }
    return;
  }

  // Else we are in normal 'ForStmt', so we can diagnose everything.
  // We only have to check cond/inc if they have changed, but 'init' needs to
  // just suppress its diagnostics if it hasn't changed.
  const ValueDecl *InitVar = checkInit();
  if (Cond.has_value())
    checkCond();
  if (Inc.has_value())
    checkInc(InitVar);
}
const ValueDecl *SemaOpenACC::ForStmtBeginChecker::checkInit() {
  if (!Init) {
    if (InitChanged) {
      SemaRef.Diag(ForLoc, diag::err_acc_loop_variable)
          << SemaRef.LoopWithoutSeqInfo.Kind;
      SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc,
                   diag::note_acc_construct_here)
          << SemaRef.LoopWithoutSeqInfo.Kind;
    }
    return nullptr;
  }

  auto DiagLoopVar = [&]() {
    if (InitChanged) {
      SemaRef.Diag(Init->getBeginLoc(), diag::err_acc_loop_variable)
          << SemaRef.LoopWithoutSeqInfo.Kind;
      SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc,
                   diag::note_acc_construct_here)
          << SemaRef.LoopWithoutSeqInfo.Kind;
    }
    return nullptr;
  };

  if (const auto *ExprTemp = dyn_cast<ExprWithCleanups>(Init))
    Init = ExprTemp->getSubExpr();
  if (const auto *E = dyn_cast<Expr>(Init))
    Init = E->IgnoreParenImpCasts();

  const ValueDecl *InitVar = nullptr;

  if (const auto *BO = dyn_cast<BinaryOperator>(Init)) {
    // Allow assignment operator here.

    if (!BO->isAssignmentOp())
      return DiagLoopVar();

    const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();

    if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS))
      InitVar = DRE->getDecl();
  } else if (const auto *DS = dyn_cast<DeclStmt>(Init)) {
    // Allow T t = <whatever>
    if (!DS->isSingleDecl())
      return DiagLoopVar();

    InitVar = dyn_cast<ValueDecl>(DS->getSingleDecl());

    // Ensure we have an initializer, unless this is a record/dependent type.

    if (InitVar) {
      if (!isa<VarDecl>(InitVar))
        return DiagLoopVar();

      if (!InitVar->getType()->isRecordType() &&
          !InitVar->getType()->isDependentType() &&
          !cast<VarDecl>(InitVar)->hasInit())
        return DiagLoopVar();
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(Init)) {
    // Allow assignment operator call.
    if (CE->getOperator() != OO_Equal)
      return DiagLoopVar();

    const Expr *LHS = CE->getArg(0)->IgnoreParenImpCasts();

    if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
      InitVar = DRE->getDecl();
    } else if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
      if (isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
        InitVar = ME->getMemberDecl();
    }
  }

  if (!InitVar)
    return DiagLoopVar();

  InitVar = cast<ValueDecl>(InitVar->getCanonicalDecl());
  QualType VarType = InitVar->getType().getNonReferenceType();

  // Since we have one, all we need to do is ensure it is the right type.
  if (!isValidLoopVariableType(VarType)) {
    if (InitChanged) {
      SemaRef.Diag(InitVar->getBeginLoc(), diag::err_acc_loop_variable_type)
          << SemaRef.LoopWithoutSeqInfo.Kind << VarType;
      SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc,
                   diag::note_acc_construct_here)
          << SemaRef.LoopWithoutSeqInfo.Kind;
    }
    return nullptr;
  }

  return InitVar;
}
void SemaOpenACC::ForStmtBeginChecker::checkCond() {
  if (!*Cond) {
    SemaRef.Diag(ForLoc, diag::err_acc_loop_terminating_condition)
        << SemaRef.LoopWithoutSeqInfo.Kind;
    SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc, diag::note_acc_construct_here)
        << SemaRef.LoopWithoutSeqInfo.Kind;
  }
  // Nothing else to do here.  we could probably do some additional work to look
  // into the termination condition, but that error-prone.  For now, we don't
  // implement anything other than 'there is a termination condition', and if
  // codegen/MLIR comes up with some necessary restrictions, we can implement
  // them here.
}

void SemaOpenACC::ForStmtBeginChecker::checkInc(const ValueDecl *Init) {

  if (!*Inc) {
    SemaRef.Diag(ForLoc, diag::err_acc_loop_not_monotonic)
        << SemaRef.LoopWithoutSeqInfo.Kind;
    SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc, diag::note_acc_construct_here)
        << SemaRef.LoopWithoutSeqInfo.Kind;
    return;
  }
  auto DiagIncVar = [this] {
    SemaRef.Diag((*Inc)->getBeginLoc(), diag::err_acc_loop_not_monotonic)
        << SemaRef.LoopWithoutSeqInfo.Kind;
    SemaRef.Diag(SemaRef.LoopWithoutSeqInfo.Loc, diag::note_acc_construct_here)
        << SemaRef.LoopWithoutSeqInfo.Kind;
    return;
  };

  if (const auto *ExprTemp = dyn_cast<ExprWithCleanups>(*Inc))
    Inc = ExprTemp->getSubExpr();
  if (const auto *E = dyn_cast<Expr>(*Inc))
    Inc = E->IgnoreParenImpCasts();

  auto getDeclFromExpr = [](const Expr *E) -> const ValueDecl * {
    E = E->IgnoreParenImpCasts();
    if (const auto *FE = dyn_cast<FullExpr>(E))
      E = FE->getSubExpr();

    E = E->IgnoreParenImpCasts();

    if (!E)
      return nullptr;
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
      return dyn_cast<ValueDecl>(DRE->getDecl());

    if (const auto *ME = dyn_cast<MemberExpr>(E))
      if (isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
        return ME->getMemberDecl();

    return nullptr;
  };

  const ValueDecl *IncVar = nullptr;

  // Here we enforce the monotonically increase/decrease:
  if (const auto *UO = dyn_cast<UnaryOperator>(*Inc)) {
    // Allow increment/decrement ops.
    if (!UO->isIncrementDecrementOp())
      return DiagIncVar();
    IncVar = getDeclFromExpr(UO->getSubExpr());
  } else if (const auto *BO = dyn_cast<BinaryOperator>(*Inc)) {
    switch (BO->getOpcode()) {
    default:
      return DiagIncVar();
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_Assign:
      // += -= *= /= should all be fine here, this should be all of the
      // 'monotonical' compound-assign ops.
      // Assignment we just give up on, we could do better, and ensure that it
      // is a binary/operator expr doing more work, but that seems like a lot
      // of work for an error prone check.
      break;
    }
    IncVar = getDeclFromExpr(BO->getLHS());
  } else if (const auto *CE = dyn_cast<CXXOperatorCallExpr>(*Inc)) {
    switch (CE->getOperator()) {
    default:
      return DiagIncVar();
    case OO_PlusPlus:
    case OO_MinusMinus:
    case OO_PlusEqual:
    case OO_MinusEqual:
    case OO_StarEqual:
    case OO_SlashEqual:
    case OO_Equal:
      // += -= *= /= should all be fine here, this should be all of the
      // 'monotonical' compound-assign ops.
      // Assignment we just give up on, we could do better, and ensure that it
      // is a binary/operator expr doing more work, but that seems like a lot
      // of work for an error prone check.
      break;
    }

    IncVar = getDeclFromExpr(CE->getArg(0));

  } else if (const auto *ME = dyn_cast<CXXMemberCallExpr>(*Inc)) {
    IncVar = getDeclFromExpr(ME->getImplicitObjectArgument());
    // We can't really do much for member expressions, other than hope they are
    // doing the right thing, so give up here.
  }

  if (!IncVar)
    return DiagIncVar();

  // InitVar shouldn't be null unless there was an error, so don't diagnose if
  // that is the case. Else we should ensure that it refers to the  loop
  // value.
  if (Init && IncVar->getCanonicalDecl() != Init->getCanonicalDecl())
    return DiagIncVar();

  return;
}

void SemaOpenACC::ActOnForStmtBegin(SourceLocation ForLoc, const Stmt *OldFirst,
                                    const Stmt *First, const Stmt *OldSecond,
                                    const Stmt *Second, const Stmt *OldThird,
                                    const Stmt *Third) {
  if (!getLangOpts().OpenACC)
    return;

  std::optional<const Stmt *> S;
  if (OldSecond == Second)
    S = std::nullopt;
  else
    S = Second;
  std::optional<const Stmt *> T;
  if (OldThird == Third)
    S = std::nullopt;
  else
    S = Third;

  bool InitChanged = false;
  if (OldFirst != First) {
    InitChanged = true;

    // VarDecls are always rebuild because they are dependent, so we can do a
    // little work to suppress some of the double checking based on whether the
    // type is instantiation dependent.
    QualType OldVDTy;
    QualType NewVDTy;
    if (const auto *DS = dyn_cast<DeclStmt>(OldFirst))
      if (const VarDecl *VD = dyn_cast_if_present<VarDecl>(
              DS->isSingleDecl() ? DS->getSingleDecl() : nullptr))
        OldVDTy = VD->getType();
    if (const auto *DS = dyn_cast<DeclStmt>(First))
      if (const VarDecl *VD = dyn_cast_if_present<VarDecl>(
              DS->isSingleDecl() ? DS->getSingleDecl() : nullptr))
        NewVDTy = VD->getType();

    if (!OldVDTy.isNull() && !NewVDTy.isNull())
      InitChanged = OldVDTy->isInstantiationDependentType() !=
                    NewVDTy->isInstantiationDependentType();
  }

  ForStmtBeginChecker FSBC{*this, ForLoc, First, InitChanged, S, T};
  if (!LoopInfo.TopLevelLoopSeen) {
    FSBC.check();
  }

  ForStmtBeginHelper(ForLoc, FSBC);
}

void SemaOpenACC::ActOnForStmtBegin(SourceLocation ForLoc, const Stmt *First,
                                    const Stmt *Second, const Stmt *Third) {
  if (!getLangOpts().OpenACC)
    return;

  ForStmtBeginChecker FSBC{*this,  ForLoc, First, /*InitChanged=*/true,
                           Second, Third};
  if (!LoopInfo.TopLevelLoopSeen) {
    FSBC.check();
  }

  ForStmtBeginHelper(ForLoc, FSBC);
}

void SemaOpenACC::ActOnRangeForStmtBegin(SourceLocation ForLoc,
                                         const Stmt *OldRangeFor,
                                         const Stmt *RangeFor) {
  if (!getLangOpts().OpenACC)
    return;

  std::optional<const CXXForRangeStmt *> RF;

  if (OldRangeFor == RangeFor)
    RF = std::nullopt;
  else
    RF = cast<CXXForRangeStmt>(RangeFor);

  ForStmtBeginChecker FSBC{*this, ForLoc, RF};
  if (!LoopInfo.TopLevelLoopSeen) {
    FSBC.check();
  }
  ForStmtBeginHelper(ForLoc, FSBC);
}

void SemaOpenACC::ActOnRangeForStmtBegin(SourceLocation ForLoc,
                                         const Stmt *RangeFor) {
  if (!getLangOpts().OpenACC)
    return;

  ForStmtBeginChecker FSBC{*this, ForLoc, cast<CXXForRangeStmt>(RangeFor)};
  if (!LoopInfo.TopLevelLoopSeen) {
    FSBC.check();
  }
  ForStmtBeginHelper(ForLoc, FSBC);
}

namespace {
SourceLocation FindInterveningCodeInLoop(const Stmt *CurStmt) {
  // We should diagnose on anything except `CompoundStmt`, `NullStmt`,
  // `ForStmt`, `CXXForRangeStmt`, since those are legal, and `WhileStmt` and
  // `DoStmt`, as those are caught as a violation elsewhere.
  // For `CompoundStmt` we need to search inside of it.
  if (!CurStmt ||
      isa<ForStmt, NullStmt, ForStmt, CXXForRangeStmt, WhileStmt, DoStmt>(
          CurStmt))
    return SourceLocation{};

  // Any other construct is an error anyway, so it has already been diagnosed.
  if (isa<OpenACCConstructStmt>(CurStmt))
    return SourceLocation{};

  // Search inside the compound statement, this allows for arbitrary nesting
  // of compound statements, as long as there isn't any code inside.
  if (const auto *CS = dyn_cast<CompoundStmt>(CurStmt)) {
    for (const auto *ChildStmt : CS->children()) {
      SourceLocation ChildStmtLoc = FindInterveningCodeInLoop(ChildStmt);
      if (ChildStmtLoc.isValid())
        return ChildStmtLoc;
    }
    // Empty/not invalid compound statements are legal.
    return SourceLocation{};
  }
  return CurStmt->getBeginLoc();
}
} // namespace

void SemaOpenACC::ActOnForStmtEnd(SourceLocation ForLoc, StmtResult Body) {
  if (!getLangOpts().OpenACC)
    return;

  // Set this to 'true' so if we find another one at this level we can diagnose.
  LoopInfo.CurLevelHasLoopAlready = true;

  if (!Body.isUsable())
    return;

  bool IsActiveCollapse = CollapseInfo.CurCollapseCount &&
                          *CollapseInfo.CurCollapseCount > 0 &&
                          !CollapseInfo.ActiveCollapse->hasForce();
  bool IsActiveTile = TileInfo.CurTileCount && *TileInfo.CurTileCount > 0;

  if (IsActiveCollapse || IsActiveTile) {
    SourceLocation OtherStmtLoc = FindInterveningCodeInLoop(Body.get());

    if (OtherStmtLoc.isValid() && IsActiveCollapse) {
      Diag(OtherStmtLoc, diag::err_acc_intervening_code)
          << OpenACCClauseKind::Collapse << CollapseInfo.DirectiveKind;
      Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
           diag::note_acc_active_clause_here)
          << OpenACCClauseKind::Collapse;
    }

    if (OtherStmtLoc.isValid() && IsActiveTile) {
      Diag(OtherStmtLoc, diag::err_acc_intervening_code)
          << OpenACCClauseKind::Tile << TileInfo.DirectiveKind;
      Diag(TileInfo.ActiveTile->getBeginLoc(),
           diag::note_acc_active_clause_here)
          << OpenACCClauseKind::Tile;
    }
  }
}

namespace {
// Get a list of clause Kinds for diagnosing a list, joined by a commas and an
// 'or'.
std::string GetListOfClauses(llvm::ArrayRef<OpenACCClauseKind> Clauses) {
  assert(!Clauses.empty() && "empty clause list not supported");

  std::string Output;
  llvm::raw_string_ostream OS{Output};

  if (Clauses.size() == 1) {
    OS << '\'' << Clauses[0] << '\'';
    return Output;
  }

  llvm::ArrayRef<OpenACCClauseKind> AllButLast{Clauses.begin(),
                                               Clauses.end() - 1};

  llvm::interleave(
      AllButLast, [&](OpenACCClauseKind K) { OS << '\'' << K << '\''; },
      [&] { OS << ", "; });

  OS << " or \'" << Clauses.back() << '\'';
  return Output;
}
} // namespace

bool SemaOpenACC::ActOnStartStmtDirective(
    OpenACCDirectiveKind K, SourceLocation StartLoc,
    ArrayRef<const OpenACCClause *> Clauses) {
  SemaRef.DiscardCleanupsInEvaluationContext();
  SemaRef.PopExpressionEvaluationContext();

  // OpenACC 3.3 2.9.1:
  // Intervening code must not contain other OpenACC directives or calls to API
  // routines.
  //
  // ALL constructs are ill-formed if there is an active 'collapse'
  if (CollapseInfo.CurCollapseCount && *CollapseInfo.CurCollapseCount > 0) {
    Diag(StartLoc, diag::err_acc_invalid_in_loop)
        << /*OpenACC Construct*/ 0 << CollapseInfo.DirectiveKind
        << OpenACCClauseKind::Collapse << K;
    assert(CollapseInfo.ActiveCollapse && "Collapse count without object?");
    Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
         diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Collapse;
  }
  if (TileInfo.CurTileCount && *TileInfo.CurTileCount > 0) {
    Diag(StartLoc, diag::err_acc_invalid_in_loop)
        << /*OpenACC Construct*/ 0 << TileInfo.DirectiveKind
        << OpenACCClauseKind::Tile << K;
    assert(TileInfo.ActiveTile && "Tile count without object?");
    Diag(TileInfo.ActiveTile->getBeginLoc(), diag::note_acc_active_clause_here)
        << OpenACCClauseKind::Tile;
  }

  // OpenACC3.3 2.6.5: At least one copy, copyin, copyout, create, no_create,
  // present, deviceptr, attach, or default clause must appear on a 'data'
  // construct.
  if (K == OpenACCDirectiveKind::Data &&
      llvm::find_if(Clauses,
                    llvm::IsaPred<OpenACCCopyClause, OpenACCCopyInClause,
                                  OpenACCCopyOutClause, OpenACCCreateClause,
                                  OpenACCNoCreateClause, OpenACCPresentClause,
                                  OpenACCDevicePtrClause, OpenACCAttachClause,
                                  OpenACCDefaultClause>) == Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K
           << GetListOfClauses(
                  {OpenACCClauseKind::Copy, OpenACCClauseKind::CopyIn,
                   OpenACCClauseKind::CopyOut, OpenACCClauseKind::Create,
                   OpenACCClauseKind::NoCreate, OpenACCClauseKind::Present,
                   OpenACCClauseKind::DevicePtr, OpenACCClauseKind::Attach,
                   OpenACCClauseKind::Default});

  // OpenACC3.3 2.6.6: At least one copyin, create, or attach clause must appear
  // on an enter data directive.
  if (K == OpenACCDirectiveKind::EnterData &&
      llvm::find_if(Clauses,
                    llvm::IsaPred<OpenACCCopyInClause, OpenACCCreateClause,
                                  OpenACCAttachClause>) == Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K
           << GetListOfClauses({
                  OpenACCClauseKind::CopyIn,
                  OpenACCClauseKind::Create,
                  OpenACCClauseKind::Attach,
              });
  // OpenACC3.3 2.6.6: At least one copyout, delete, or detach clause must
  // appear on an exit data directive.
  if (K == OpenACCDirectiveKind::ExitData &&
      llvm::find_if(Clauses,
                    llvm::IsaPred<OpenACCCopyOutClause, OpenACCDeleteClause,
                                  OpenACCDetachClause>) == Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K
           << GetListOfClauses({
                  OpenACCClauseKind::CopyOut,
                  OpenACCClauseKind::Delete,
                  OpenACCClauseKind::Detach,
              });

  // OpenACC3.3 2.8: At least 'one use_device' clause must appear.
  if (K == OpenACCDirectiveKind::HostData &&
      llvm::find_if(Clauses, llvm::IsaPred<OpenACCUseDeviceClause>) ==
          Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K << GetListOfClauses({OpenACCClauseKind::UseDevice});

  // OpenACC3.3 2.14.3: At least one default_async, device_num, or device_type
  // clause must appear.
  if (K == OpenACCDirectiveKind::Set &&
      llvm::find_if(
          Clauses,
          llvm::IsaPred<OpenACCDefaultAsyncClause, OpenACCDeviceNumClause,
                        OpenACCDeviceTypeClause, OpenACCIfClause>) ==
          Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K
           << GetListOfClauses({OpenACCClauseKind::DefaultAsync,
                                OpenACCClauseKind::DeviceNum,
                                OpenACCClauseKind::DeviceType,
                                OpenACCClauseKind::If});

  // OpenACC3.3 2.14.4: At least one self, host, or device clause must appear on
  // an update directive.
  if (K == OpenACCDirectiveKind::Update &&
      llvm::find_if(Clauses, llvm::IsaPred<OpenACCSelfClause, OpenACCHostClause,
                                           OpenACCDeviceClause>) ==
          Clauses.end())
    return Diag(StartLoc, diag::err_acc_construct_one_clause_of)
           << K
           << GetListOfClauses({OpenACCClauseKind::Self,
                                OpenACCClauseKind::Host,
                                OpenACCClauseKind::Device});

  return diagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/true);
}

StmtResult SemaOpenACC::ActOnEndStmtDirective(
    OpenACCDirectiveKind K, SourceLocation StartLoc, SourceLocation DirLoc,
    SourceLocation LParenLoc, SourceLocation MiscLoc, ArrayRef<Expr *> Exprs,
    SourceLocation RParenLoc, SourceLocation EndLoc,
    ArrayRef<OpenACCClause *> Clauses, StmtResult AssocStmt) {
  switch (K) {
  default:
    return StmtEmpty();
  case OpenACCDirectiveKind::Invalid:
    return StmtError();
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels: {
    return OpenACCComputeConstruct::Create(
        getASTContext(), K, StartLoc, DirLoc, EndLoc, Clauses,
        AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop: {
    return OpenACCCombinedConstruct::Create(
        getASTContext(), K, StartLoc, DirLoc, EndLoc, Clauses,
        AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  case OpenACCDirectiveKind::Loop: {
    return OpenACCLoopConstruct::Create(
        getASTContext(), ActiveComputeConstructInfo.Kind, StartLoc, DirLoc,
        EndLoc, Clauses, AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  case OpenACCDirectiveKind::Data: {
    return OpenACCDataConstruct::Create(
        getASTContext(), StartLoc, DirLoc, EndLoc, Clauses,
        AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  case OpenACCDirectiveKind::EnterData: {
    return OpenACCEnterDataConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                             EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::ExitData: {
    return OpenACCExitDataConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                            EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::HostData: {
    return OpenACCHostDataConstruct::Create(
        getASTContext(), StartLoc, DirLoc, EndLoc, Clauses,
        AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  case OpenACCDirectiveKind::Wait: {
    return OpenACCWaitConstruct::Create(
        getASTContext(), StartLoc, DirLoc, LParenLoc, Exprs.front(), MiscLoc,
        Exprs.drop_front(), RParenLoc, EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::Init: {
    return OpenACCInitConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                        EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::Shutdown: {
    return OpenACCShutdownConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                            EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::Set: {
    return OpenACCSetConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                       EndLoc, Clauses);
  }
  case OpenACCDirectiveKind::Update: {
    return OpenACCUpdateConstruct::Create(getASTContext(), StartLoc, DirLoc,
                                          EndLoc, Clauses);
  }
  }
  llvm_unreachable("Unhandled case in directive handling?");
}

StmtResult SemaOpenACC::ActOnAssociatedStmt(
    SourceLocation DirectiveLoc, OpenACCDirectiveKind K,
    ArrayRef<const OpenACCClause *> Clauses, StmtResult AssocStmt) {
  switch (K) {
  default:
    llvm_unreachable("Unimplemented associated statement application");
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
  case OpenACCDirectiveKind::Wait:
  case OpenACCDirectiveKind::Init:
  case OpenACCDirectiveKind::Shutdown:
  case OpenACCDirectiveKind::Set:
    llvm_unreachable(
        "these don't have associated statements, so shouldn't get here");
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
  case OpenACCDirectiveKind::Data:
  case OpenACCDirectiveKind::HostData:
    // There really isn't any checking here that could happen. As long as we
    // have a statement to associate, this should be fine.
    // OpenACC 3.3 Section 6:
    // Structured Block: in C or C++, an executable statement, possibly
    // compound, with a single entry at the top and a single exit at the
    // bottom.
    // FIXME: Should we reject DeclStmt's here? The standard isn't clear, and
    // an interpretation of it is to allow this and treat the initializer as
    // the 'structured block'.
    return AssocStmt;
  case OpenACCDirectiveKind::Loop:
  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop:
    if (!AssocStmt.isUsable())
      return StmtError();

    if (!isa<CXXForRangeStmt, ForStmt>(AssocStmt.get())) {
      Diag(AssocStmt.get()->getBeginLoc(), diag::err_acc_loop_not_for_loop)
          << K;
      Diag(DirectiveLoc, diag::note_acc_construct_here) << K;
      return StmtError();
    }

    if (!CollapseInfo.CollapseDepthSatisfied || !TileInfo.TileDepthSatisfied) {
      if (!CollapseInfo.CollapseDepthSatisfied) {
        Diag(DirectiveLoc, diag::err_acc_insufficient_loops)
            << OpenACCClauseKind::Collapse;
        assert(CollapseInfo.ActiveCollapse && "Collapse count without object?");
        Diag(CollapseInfo.ActiveCollapse->getBeginLoc(),
             diag::note_acc_active_clause_here)
            << OpenACCClauseKind::Collapse;
      }

      if (!TileInfo.TileDepthSatisfied) {
        Diag(DirectiveLoc, diag::err_acc_insufficient_loops)
            << OpenACCClauseKind::Tile;
        assert(TileInfo.ActiveTile && "Collapse count without object?");
        Diag(TileInfo.ActiveTile->getBeginLoc(),
             diag::note_acc_active_clause_here)
            << OpenACCClauseKind::Tile;
      }
      return StmtError();
    }

    return AssocStmt.get();
  }
  llvm_unreachable("Invalid associated statement application");
}

bool SemaOpenACC::ActOnStartDeclDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc) {
  // OpenCC3.3 2.1 (line 889)
  // A program must not depend on the order of evaluation of expressions in
  // clause arguments or on any side effects of the evaluations.
  SemaRef.DiscardCleanupsInEvaluationContext();
  SemaRef.PopExpressionEvaluationContext();
  return diagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/false);
}

DeclGroupRef SemaOpenACC::ActOnEndDeclDirective() { return DeclGroupRef{}; }

ExprResult
SemaOpenACC::BuildOpenACCAsteriskSizeExpr(SourceLocation AsteriskLoc) {
  return OpenACCAsteriskSizeExpr::Create(getASTContext(), AsteriskLoc);
}

ExprResult
SemaOpenACC::ActOnOpenACCAsteriskSizeExpr(SourceLocation AsteriskLoc) {
  return BuildOpenACCAsteriskSizeExpr(AsteriskLoc);
}
