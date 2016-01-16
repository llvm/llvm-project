//===--- OpenMPKinds.cpp - Token Kinds Support ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements the OpenMP enum and support functions.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;

OpenMPDirectiveKind clang::getOpenMPDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<OpenMPDirectiveKind>(Str)
#define OPENMP_DIRECTIVE(Name) .Case(#Name, OMPD_##Name)
#define OPENMP_DIRECTIVE_EXT(Name, Str) .Case(Str, OMPD_##Name)
#include "clang/Basic/OpenMPKinds.def"
      .Default(OMPD_unknown);
}

const char *clang::getOpenMPDirectiveName(OpenMPDirectiveKind Kind) {
  assert(Kind <= OMPD_unknown);
  switch (Kind) {
  case OMPD_unknown:
    return "unknown";
#define OPENMP_DIRECTIVE(Name)                                                 \
  case OMPD_##Name:                                                            \
    return #Name;
#define OPENMP_DIRECTIVE_EXT(Name, Str)                                        \
  case OMPD_##Name:                                                            \
    return Str;
#include "clang/Basic/OpenMPKinds.def"
    break;
  }
  llvm_unreachable("Invalid OpenMP directive kind");
}

OpenMPClauseKind clang::getOpenMPClauseKind(StringRef Str) {
  // 'flush' clause cannot be specified explicitly, because this is an implicit
  // clause for 'flush' directive. If the 'flush' clause is explicitly specified
  // the Parser should generate a warning about extra tokens at the end of the
  // directive.
  if (Str == "flush")
    return OMPC_unknown;
  return llvm::StringSwitch<OpenMPClauseKind>(Str)
#define OPENMP_CLAUSE(Name, Class) .Case(#Name, OMPC_##Name)
#include "clang/Basic/OpenMPKinds.def"
      .Default(OMPC_unknown);
}

const char *clang::getOpenMPClauseName(OpenMPClauseKind Kind) {
  assert(Kind <= OMPC_unknown);
  switch (Kind) {
  case OMPC_unknown:
    return "unknown";
#define OPENMP_CLAUSE(Name, Class)                                             \
  case OMPC_##Name:                                                            \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
  case OMPC_threadprivate:
    return "threadprivate or thread local";
  }
  llvm_unreachable("Invalid OpenMP clause kind");
}

unsigned clang::getOpenMPSimpleClauseType(OpenMPClauseKind Kind,
                                          StringRef Str) {
  switch (Kind) {
  case OMPC_default:
    return llvm::StringSwitch<OpenMPDefaultClauseKind>(Str)
#define OPENMP_DEFAULT_KIND(Name) .Case(#Name, OMPC_DEFAULT_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_DEFAULT_unknown);
  case OMPC_proc_bind:
    return llvm::StringSwitch<OpenMPProcBindClauseKind>(Str)
#define OPENMP_PROC_BIND_KIND(Name) .Case(#Name, OMPC_PROC_BIND_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_PROC_BIND_unknown);
  case OMPC_schedule:
    return llvm::StringSwitch<unsigned>(Str)
#define OPENMP_SCHEDULE_KIND(Name)                                             \
  .Case(#Name, static_cast<unsigned>(OMPC_SCHEDULE_##Name))
#define OPENMP_SCHEDULE_MODIFIER(Name)                                         \
  .Case(#Name, static_cast<unsigned>(OMPC_SCHEDULE_MODIFIER_##Name))
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_SCHEDULE_unknown);
  case OMPC_depend:
    return llvm::StringSwitch<OpenMPDependClauseKind>(Str)
#define OPENMP_DEPEND_KIND(Name) .Case(#Name, OMPC_DEPEND_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_DEPEND_unknown);
  case OMPC_linear:
    return llvm::StringSwitch<OpenMPLinearClauseKind>(Str)
#define OPENMP_LINEAR_KIND(Name) .Case(#Name, OMPC_LINEAR_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_LINEAR_unknown);
  case OMPC_map:
    return llvm::StringSwitch<OpenMPMapClauseKind>(Str)
#define OPENMP_MAP_KIND(Name) .Case(#Name, OMPC_MAP_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_MAP_unknown);
  case OMPC_dist_schedule:
    return llvm::StringSwitch<OpenMPDistScheduleClauseKind>(Str)
#define OPENMP_DIST_SCHEDULE_KIND(Name) .Case(#Name, OMPC_DIST_SCHEDULE_##Name)
#include "clang/Basic/OpenMPKinds.def"
        .Default(OMPC_DIST_SCHEDULE_unknown);
  case OMPC_unknown:
  case OMPC_threadprivate:
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
    break;
  }
  llvm_unreachable("Invalid OpenMP simple clause kind");
}

const char *clang::getOpenMPSimpleClauseTypeName(OpenMPClauseKind Kind,
                                                 unsigned Type) {
  switch (Kind) {
  case OMPC_default:
    switch (Type) {
    case OMPC_DEFAULT_unknown:
      return "unknown";
#define OPENMP_DEFAULT_KIND(Name)                                              \
  case OMPC_DEFAULT_##Name:                                                    \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'default' clause type");
  case OMPC_proc_bind:
    switch (Type) {
    case OMPC_PROC_BIND_unknown:
      return "unknown";
#define OPENMP_PROC_BIND_KIND(Name)                                            \
  case OMPC_PROC_BIND_##Name:                                                  \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'proc_bind' clause type");
  case OMPC_schedule:
    switch (Type) {
    case OMPC_SCHEDULE_unknown:
    case OMPC_SCHEDULE_MODIFIER_last:
      return "unknown";
#define OPENMP_SCHEDULE_KIND(Name)                                             \
    case OMPC_SCHEDULE_##Name:                                                 \
      return #Name;
#define OPENMP_SCHEDULE_MODIFIER(Name)                                         \
    case OMPC_SCHEDULE_MODIFIER_##Name:                                        \
      return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'schedule' clause type");
  case OMPC_depend:
    switch (Type) {
    case OMPC_DEPEND_unknown:
      return "unknown";
#define OPENMP_DEPEND_KIND(Name)                                             \
  case OMPC_DEPEND_##Name:                                                   \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'depend' clause type");
  case OMPC_linear:
    switch (Type) {
    case OMPC_LINEAR_unknown:
      return "unknown";
#define OPENMP_LINEAR_KIND(Name)                                             \
  case OMPC_LINEAR_##Name:                                                   \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'linear' clause type");
  case OMPC_map:
    switch (Type) {
    case OMPC_MAP_unknown:
      return "unknown";
#define OPENMP_MAP_KIND(Name)                                                \
  case OMPC_MAP_##Name:                                                      \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    llvm_unreachable("Invalid OpenMP 'map' clause type");
  case OMPC_dist_schedule:
    switch (Type) {
    case OMPC_DIST_SCHEDULE_unknown:
      return "unknown";
#define OPENMP_DIST_SCHEDULE_KIND(Name)                                      \
  case OMPC_DIST_SCHEDULE_##Name:                                            \
    return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'dist_schedule' clause type");
  case OMPC_unknown:
  case OMPC_threadprivate:
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
    break;
  }
  llvm_unreachable("Invalid OpenMP simple clause kind");
}

bool clang::isAllowedClauseForDirective(OpenMPDirectiveKind DKind,
                                        OpenMPClauseKind CKind) {
  assert(DKind <= OMPD_unknown);
  assert(CKind <= OMPC_unknown);
  switch (DKind) {
  case OMPD_parallel:
    switch (CKind) {
#define OPENMP_PARALLEL_CLAUSE(Name)                                           \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_simd:
    switch (CKind) {
#define OPENMP_SIMD_CLAUSE(Name)                                               \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_for:
    switch (CKind) {
#define OPENMP_FOR_CLAUSE(Name)                                                \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_for_simd:
    switch (CKind) {
#define OPENMP_FOR_SIMD_CLAUSE(Name)                                           \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_sections:
    switch (CKind) {
#define OPENMP_SECTIONS_CLAUSE(Name)                                           \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_single:
    switch (CKind) {
#define OPENMP_SINGLE_CLAUSE(Name)                                             \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_parallel_for:
    switch (CKind) {
#define OPENMP_PARALLEL_FOR_CLAUSE(Name)                                       \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_parallel_for_simd:
    switch (CKind) {
#define OPENMP_PARALLEL_FOR_SIMD_CLAUSE(Name)                                  \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_parallel_sections:
    switch (CKind) {
#define OPENMP_PARALLEL_SECTIONS_CLAUSE(Name)                                  \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_task:
    switch (CKind) {
#define OPENMP_TASK_CLAUSE(Name)                                               \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_flush:
    return CKind == OMPC_flush;
    break;
  case OMPD_atomic:
    switch (CKind) {
#define OPENMP_ATOMIC_CLAUSE(Name)                                             \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_target:
    switch (CKind) {
#define OPENMP_TARGET_CLAUSE(Name)                                             \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_target_data:
    switch (CKind) {
#define OPENMP_TARGET_DATA_CLAUSE(Name)                                        \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_teams:
    switch (CKind) {
#define OPENMP_TEAMS_CLAUSE(Name)                                              \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_cancel:
    switch (CKind) {
#define OPENMP_CANCEL_CLAUSE(Name)                                             \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_ordered:
    switch (CKind) {
#define OPENMP_ORDERED_CLAUSE(Name)                                            \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_taskloop:
    switch (CKind) {
#define OPENMP_TASKLOOP_CLAUSE(Name)                                           \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_taskloop_simd:
    switch (CKind) {
#define OPENMP_TASKLOOP_SIMD_CLAUSE(Name)                                      \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_critical:
    switch (CKind) {
#define OPENMP_CRITICAL_CLAUSE(Name)                                           \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_distribute:
    switch (CKind) {
#define OPENMP_DISTRIBUTE_CLAUSE(Name)                                         \
  case OMPC_##Name:                                                            \
    return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_unknown:
  case OMPD_threadprivate:
  case OMPD_section:
  case OMPD_master:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_cancellation_point:
    break;
  }
  return false;
}

bool clang::isOpenMPLoopDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_simd || DKind == OMPD_for || DKind == OMPD_for_simd ||
         DKind == OMPD_parallel_for || DKind == OMPD_parallel_for_simd ||
         DKind == OMPD_taskloop ||
         DKind == OMPD_taskloop_simd ||
         DKind == OMPD_distribute; // TODO add next directives.
}

bool clang::isOpenMPWorksharingDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_for || DKind == OMPD_for_simd ||
         DKind == OMPD_sections || DKind == OMPD_section ||
         DKind == OMPD_single || DKind == OMPD_parallel_for ||
         DKind == OMPD_parallel_for_simd ||
         DKind == OMPD_parallel_sections; // TODO add next directives.
}

bool clang::isOpenMPTaskLoopDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_taskloop || DKind == OMPD_taskloop_simd;
}

bool clang::isOpenMPParallelDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_parallel || DKind == OMPD_parallel_for ||
         DKind == OMPD_parallel_for_simd ||
         DKind == OMPD_parallel_sections; // TODO add next directives.
}

bool clang::isOpenMPTargetDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_target; // TODO add next directives.
}

bool clang::isOpenMPTeamsDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_teams; // TODO add next directives.
}

bool clang::isOpenMPSimdDirective(OpenMPDirectiveKind DKind) {
  return DKind == OMPD_simd || DKind == OMPD_for_simd ||
         DKind == OMPD_parallel_for_simd ||
         DKind == OMPD_taskloop_simd; // TODO add next directives.
}

bool clang::isOpenMPDistributeDirective(OpenMPDirectiveKind Kind) {
  return Kind == OMPD_distribute; // TODO add next directives.
}

bool clang::isOpenMPPrivate(OpenMPClauseKind Kind) {
  return Kind == OMPC_private || Kind == OMPC_firstprivate ||
         Kind == OMPC_lastprivate || Kind == OMPC_linear ||
         Kind == OMPC_reduction; // TODO add next clauses like 'reduction'.
}

bool clang::isOpenMPThreadPrivate(OpenMPClauseKind Kind) {
  return Kind == OMPC_threadprivate || Kind == OMPC_copyin;
}

