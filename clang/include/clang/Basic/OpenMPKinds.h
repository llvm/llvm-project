//===--- OpenMPKinds.h - OpenMP enums ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some OpenMP-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENMPKINDS_H
#define LLVM_CLANG_BASIC_OPENMPKINDS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace clang {

/// OpenMP context selector sets.
enum OpenMPContextSelectorSetKind {
#define OPENMP_CONTEXT_SELECTOR_SET(Name) OMP_CTX_SET_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMP_CTX_SET_unknown,
};

/// OpenMP context selectors.
enum OpenMPContextSelectorKind {
#define OPENMP_CONTEXT_SELECTOR(Name) OMP_CTX_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMP_CTX_unknown,
};

OpenMPContextSelectorSetKind getOpenMPContextSelectorSet(llvm::StringRef Str);
llvm::StringRef
getOpenMPContextSelectorSetName(OpenMPContextSelectorSetKind Kind);
OpenMPContextSelectorKind getOpenMPContextSelector(llvm::StringRef Str);
llvm::StringRef getOpenMPContextSelectorName(OpenMPContextSelectorKind Kind);

/// Struct to store the context selectors info.
template <typename VectorType, typename ScoreT> struct OpenMPCtxSelectorData {
  OpenMPContextSelectorSetKind CtxSet = OMP_CTX_SET_unknown;
  OpenMPContextSelectorKind Ctx = OMP_CTX_unknown;
  ScoreT Score;
  VectorType Names;
  explicit OpenMPCtxSelectorData() = default;
  explicit OpenMPCtxSelectorData(OpenMPContextSelectorSetKind CtxSet,
                                 OpenMPContextSelectorKind Ctx,
                                 const ScoreT &Score, VectorType &&Names)
      : CtxSet(CtxSet), Ctx(Ctx), Score(Score), Names(Names) {}
  template <typename U>
  explicit OpenMPCtxSelectorData(OpenMPContextSelectorSetKind CtxSet,
                                 OpenMPContextSelectorKind Ctx,
                                 const ScoreT &Score, const U &Names)
      : CtxSet(CtxSet), Ctx(Ctx), Score(Score),
        Names(Names.begin(), Names.end()) {}
};

/// OpenMP directives.
using OpenMPDirectiveKind = llvm::omp::Directive;

/// OpenMP clauses.
enum OpenMPClauseKind {
#define OPENMP_CLAUSE(Name, Class) \
  OMPC_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_threadprivate,
  OMPC_uniform,
  OMPC_device_type,
  OMPC_match,
  OMPC_unknown
};

/// OpenMP attributes for 'default' clause.
enum OpenMPDefaultClauseKind {
#define OPENMP_DEFAULT_KIND(Name) \
  OMPC_DEFAULT_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEFAULT_unknown
};

/// OpenMP attributes for 'schedule' clause.
enum OpenMPScheduleClauseKind {
#define OPENMP_SCHEDULE_KIND(Name) \
  OMPC_SCHEDULE_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_SCHEDULE_unknown
};

/// OpenMP modifiers for 'schedule' clause.
enum OpenMPScheduleClauseModifier {
  OMPC_SCHEDULE_MODIFIER_unknown = OMPC_SCHEDULE_unknown,
#define OPENMP_SCHEDULE_MODIFIER(Name) \
  OMPC_SCHEDULE_MODIFIER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_SCHEDULE_MODIFIER_last
};

/// OpenMP attributes for 'depend' clause.
enum OpenMPDependClauseKind {
#define OPENMP_DEPEND_KIND(Name) \
  OMPC_DEPEND_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEPEND_unknown
};

/// OpenMP attributes for 'linear' clause.
enum OpenMPLinearClauseKind {
#define OPENMP_LINEAR_KIND(Name) \
  OMPC_LINEAR_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_LINEAR_unknown
};

/// OpenMP mapping kind for 'map' clause.
enum OpenMPMapClauseKind {
#define OPENMP_MAP_KIND(Name) \
  OMPC_MAP_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_MAP_unknown
};

/// OpenMP modifier kind for 'map' clause.
enum OpenMPMapModifierKind {
  OMPC_MAP_MODIFIER_unknown = OMPC_MAP_unknown,
#define OPENMP_MAP_MODIFIER_KIND(Name) \
  OMPC_MAP_MODIFIER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_MAP_MODIFIER_last
};

/// OpenMP modifier kind for 'to' clause.
enum OpenMPToModifierKind {
#define OPENMP_TO_MODIFIER_KIND(Name) \
  OMPC_TO_MODIFIER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_TO_MODIFIER_unknown
};

/// OpenMP modifier kind for 'from' clause.
enum OpenMPFromModifierKind {
#define OPENMP_FROM_MODIFIER_KIND(Name) \
  OMPC_FROM_MODIFIER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_FROM_MODIFIER_unknown
};

/// OpenMP attributes for 'dist_schedule' clause.
enum OpenMPDistScheduleClauseKind {
#define OPENMP_DIST_SCHEDULE_KIND(Name) OMPC_DIST_SCHEDULE_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DIST_SCHEDULE_unknown
};

/// OpenMP attributes for 'defaultmap' clause.
enum OpenMPDefaultmapClauseKind {
#define OPENMP_DEFAULTMAP_KIND(Name) \
  OMPC_DEFAULTMAP_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEFAULTMAP_unknown
};

/// OpenMP modifiers for 'defaultmap' clause.
enum OpenMPDefaultmapClauseModifier {
  OMPC_DEFAULTMAP_MODIFIER_unknown = OMPC_DEFAULTMAP_unknown,
#define OPENMP_DEFAULTMAP_MODIFIER(Name) \
  OMPC_DEFAULTMAP_MODIFIER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEFAULTMAP_MODIFIER_last
};

/// OpenMP attributes for 'atomic_default_mem_order' clause.
enum OpenMPAtomicDefaultMemOrderClauseKind {
#define OPENMP_ATOMIC_DEFAULT_MEM_ORDER_KIND(Name)  \
  OMPC_ATOMIC_DEFAULT_MEM_ORDER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_ATOMIC_DEFAULT_MEM_ORDER_unknown
};

/// OpenMP device type for 'device_type' clause.
enum OpenMPDeviceType {
#define OPENMP_DEVICE_TYPE_KIND(Name) \
  OMPC_DEVICE_TYPE_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEVICE_TYPE_unknown
};

/// OpenMP 'lastprivate' clause modifier.
enum OpenMPLastprivateModifier {
#define OPENMP_LASTPRIVATE_KIND(Name) OMPC_LASTPRIVATE_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_LASTPRIVATE_unknown,
};

/// OpenMP attributes for 'order' clause.
enum OpenMPOrderClauseKind {
#define OPENMP_ORDER_KIND(Name) OMPC_ORDER_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_ORDER_unknown,
};

/// Scheduling data for loop-based OpenMP directives.
struct OpenMPScheduleTy final {
  OpenMPScheduleClauseKind Schedule = OMPC_SCHEDULE_unknown;
  OpenMPScheduleClauseModifier M1 = OMPC_SCHEDULE_MODIFIER_unknown;
  OpenMPScheduleClauseModifier M2 = OMPC_SCHEDULE_MODIFIER_unknown;
};

OpenMPClauseKind getOpenMPClauseKind(llvm::StringRef Str);
const char *getOpenMPClauseName(OpenMPClauseKind Kind);

unsigned getOpenMPSimpleClauseType(OpenMPClauseKind Kind, llvm::StringRef Str);
const char *getOpenMPSimpleClauseTypeName(OpenMPClauseKind Kind, unsigned Type);

bool isAllowedClauseForDirective(OpenMPDirectiveKind DKind,
                                 OpenMPClauseKind CKind,
                                 unsigned OpenMPVersion);

/// Checks if the specified directive is a directive with an associated
/// loop construct.
/// \param DKind Specified directive.
/// \return true - the directive is a loop-associated directive like 'omp simd'
/// or 'omp for' directive, otherwise - false.
bool isOpenMPLoopDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a worksharing directive.
/// \param DKind Specified directive.
/// \return true - the directive is a worksharing directive like 'omp for',
/// otherwise - false.
bool isOpenMPWorksharingDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a taskloop directive.
/// \param DKind Specified directive.
/// \return true - the directive is a worksharing directive like 'omp taskloop',
/// otherwise - false.
bool isOpenMPTaskLoopDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a parallel-kind directive.
/// \param DKind Specified directive.
/// \return true - the directive is a parallel-like directive like 'omp
/// parallel', otherwise - false.
bool isOpenMPParallelDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a target code offload directive.
/// \param DKind Specified directive.
/// \return true - the directive is a target code offload directive like
/// 'omp target', 'omp target parallel', 'omp target xxx'
/// otherwise - false.
bool isOpenMPTargetExecutionDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a target data offload directive.
/// \param DKind Specified directive.
/// \return true - the directive is a target data offload directive like
/// 'omp target data', 'omp target update', 'omp target enter data',
/// 'omp target exit data'
/// otherwise - false.
bool isOpenMPTargetDataManagementDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified composite/combined directive constitutes a teams
/// directive in the outermost nest.  For example
/// 'omp teams distribute' or 'omp teams distribute parallel for'.
/// \param DKind Specified directive.
/// \return true - the directive has teams on the outermost nest, otherwise -
/// false.
bool isOpenMPNestingTeamsDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a teams-kind directive.  For example,
/// 'omp teams distribute' or 'omp target teams'.
/// \param DKind Specified directive.
/// \return true - the directive is a teams-like directive, otherwise - false.
bool isOpenMPTeamsDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a simd directive.
/// \param DKind Specified directive.
/// \return true - the directive is a simd directive like 'omp simd',
/// otherwise - false.
bool isOpenMPSimdDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified directive is a distribute directive.
/// \param DKind Specified directive.
/// \return true - the directive is a distribute-directive like 'omp
/// distribute',
/// otherwise - false.
bool isOpenMPDistributeDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified composite/combined directive constitutes a
/// distribute directive in the outermost nest.  For example,
/// 'omp distribute parallel for' or 'omp distribute'.
/// \param DKind Specified directive.
/// \return true - the directive has distribute on the outermost nest.
/// otherwise - false.
bool isOpenMPNestingDistributeDirective(OpenMPDirectiveKind DKind);

/// Checks if the specified clause is one of private clauses like
/// 'private', 'firstprivate', 'reduction' etc..
/// \param Kind Clause kind.
/// \return true - the clause is a private clause, otherwise - false.
bool isOpenMPPrivate(OpenMPClauseKind Kind);

/// Checks if the specified clause is one of threadprivate clauses like
/// 'threadprivate', 'copyin' or 'copyprivate'.
/// \param Kind Clause kind.
/// \return true - the clause is a threadprivate clause, otherwise - false.
bool isOpenMPThreadPrivate(OpenMPClauseKind Kind);

/// Checks if the specified directive kind is one of tasking directives - task,
/// taskloop, taksloop simd, master taskloop, parallel master taskloop, master
/// taskloop simd, or parallel master taskloop simd.
bool isOpenMPTaskingDirective(OpenMPDirectiveKind Kind);

/// Checks if the specified directive kind is one of the composite or combined
/// directives that need loop bound sharing across loops outlined in nested
/// functions
bool isOpenMPLoopBoundSharingDirective(OpenMPDirectiveKind Kind);

/// Return the captured regions of an OpenMP directive.
void getOpenMPCaptureRegions(
    llvm::SmallVectorImpl<OpenMPDirectiveKind> &CaptureRegions,
    OpenMPDirectiveKind DKind);
}

#endif

