//===--- OpenMPKinds.h - OpenMP enums ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines some OpenMP-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENMPKINDS_H
#define LLVM_CLANG_BASIC_OPENMPKINDS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

/// \brief OpenMP directives.
enum OpenMPDirectiveKind {
#define OPENMP_DIRECTIVE(Name) \
  OMPD_##Name,
#define OPENMP_DIRECTIVE_EXT(Name, Str) \
  OMPD_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPD_unknown
};

/// \brief OpenMP clauses.
enum OpenMPClauseKind {
#define OPENMP_CLAUSE(Name, Class) \
  OMPC_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_threadprivate,
  OMPC_unknown
};

/// \brief OpenMP attributes for 'default' clause.
enum OpenMPDefaultClauseKind {
#define OPENMP_DEFAULT_KIND(Name) \
  OMPC_DEFAULT_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEFAULT_unknown
};

/// \brief OpenMP attributes for 'proc_bind' clause.
enum OpenMPProcBindClauseKind {
#define OPENMP_PROC_BIND_KIND(Name) \
  OMPC_PROC_BIND_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_PROC_BIND_unknown
};

/// \brief OpenMP attributes for 'schedule' clause.
enum OpenMPScheduleClauseKind {
#define OPENMP_SCHEDULE_KIND(Name) \
  OMPC_SCHEDULE_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_SCHEDULE_unknown
};

/// \brief OpenMP attributes for 'depend' clause.
enum OpenMPDependClauseKind {
#define OPENMP_DEPEND_KIND(Name) \
  OMPC_DEPEND_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_DEPEND_unknown
};

/// \brief OpenMP attributes for 'linear' clause.
enum OpenMPLinearClauseKind {
#define OPENMP_LINEAR_KIND(Name) \
  OMPC_LINEAR_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_LINEAR_unknown
};

/// \brief OpenMP mapping kind for 'map' clause.
enum OpenMPMapClauseKind {
#define OPENMP_MAP_KIND(Name) \
  OMPC_MAP_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_MAP_unknown
};

OpenMPDirectiveKind getOpenMPDirectiveKind(llvm::StringRef Str);
const char *getOpenMPDirectiveName(OpenMPDirectiveKind Kind);

OpenMPClauseKind getOpenMPClauseKind(llvm::StringRef Str);
const char *getOpenMPClauseName(OpenMPClauseKind Kind);

unsigned getOpenMPSimpleClauseType(OpenMPClauseKind Kind, llvm::StringRef Str);
const char *getOpenMPSimpleClauseTypeName(OpenMPClauseKind Kind, unsigned Type);

bool isAllowedClauseForDirective(OpenMPDirectiveKind DKind,
                                 OpenMPClauseKind CKind);

/// \brief Checks if the specified directive is a directive with an associated
/// loop construct.
/// \param DKind Specified directive.
/// \return true - the directive is a loop-associated directive like 'omp simd'
/// or 'omp for' directive, otherwise - false.
bool isOpenMPLoopDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified directive is a worksharing directive.
/// \param DKind Specified directive.
/// \return true - the directive is a worksharing directive like 'omp for',
/// otherwise - false.
bool isOpenMPWorksharingDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified directive is a parallel-kind directive.
/// \param DKind Specified directive.
/// \return true - the directive is a parallel-like directive like 'omp
/// parallel', otherwise - false.
bool isOpenMPParallelDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified directive is a target-kind directive.
/// \param DKind Specified directive.
/// \return true - the directive is a target-like directive like 'omp target',
/// otherwise - false.
bool isOpenMPTargetDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified directive is a teams-kind directive.
/// \param DKind Specified directive.
/// \return true - the directive is a teams-like directive like 'omp teams',
/// otherwise - false.
bool isOpenMPTeamsDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified directive is a simd directive.
/// \param DKind Specified directive.
/// \return true - the directive is a simd directive like 'omp simd',
/// otherwise - false.
bool isOpenMPSimdDirective(OpenMPDirectiveKind DKind);

/// \brief Checks if the specified clause is one of private clauses like
/// 'private', 'firstprivate', 'reduction' etc..
/// \param Kind Clause kind.
/// \return true - the clause is a private clause, otherwise - false.
bool isOpenMPPrivate(OpenMPClauseKind Kind);

/// \brief Checks if the specified clause is one of threadprivate clauses like
/// 'threadprivate', 'copyin' or 'copyprivate'.
/// \param Kind Clause kind.
/// \return true - the clause is a threadprivate clause, otherwise - false.
bool isOpenMPThreadPrivate(OpenMPClauseKind Kind);

}

#endif

