//===-- RecordOps.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Operations on records (structs, classes, and unions).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_RECORDOPS_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_RECORDOPS_H

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"

namespace clang {
namespace dataflow {

/// Copies a record (struct, class, or union) from `Src` to `Dst`.
///
/// This performs a deep copy, i.e. it copies every field (including synthetic
/// fields) and recurses on fields of record type. It also copies properties
/// from the `RecordValue` associated with `Src` to the `RecordValue` associated
/// with `Dst` (if these `RecordValue`s exist).
///
/// If there is a `RecordValue` associated with `Dst` in the environment, this
/// function creates a new `RecordValue` and associates it with `Dst`; clients
/// need to be aware of this and must not assume that the `RecordValue`
/// associated with `Dst` remains the same after the call.
///
/// We create a new `RecordValue` rather than modifying properties on the old
/// `RecordValue` because the old `RecordValue` may be shared with other
/// `Environment`s, and we don't want changes to properties to be visible there.
///
/// Requirements:
///
///  `Src` and `Dst` must have the same canonical unqualified type.
void copyRecord(RecordStorageLocation &Src, RecordStorageLocation &Dst,
                Environment &Env);

/// Returns whether the records `Loc1` and `Loc2` are equal.
///
/// Values for `Loc1` are retrieved from `Env1`, and values for `Loc2` are
/// retrieved from `Env2`. A convenience overload retrieves values for `Loc1`
/// and `Loc2` from the same environment.
///
/// This performs a deep comparison, i.e. it compares every field (including
/// synthetic fields) and recurses on fields of record type. Fields of reference
/// type compare equal if they refer to the same storage location. If
/// `RecordValue`s are associated with `Loc1` and Loc2`, it also compares the
/// properties on those `RecordValue`s.
///
/// Note on how to interpret the result:
/// - If this returns true, the records are guaranteed to be equal at runtime.
/// - If this returns false, the records may still be equal at runtime; our
///   analysis merely cannot guarantee that they will be equal.
///
/// Requirements:
///
///  `Src` and `Dst` must have the same canonical unqualified type.
bool recordsEqual(const RecordStorageLocation &Loc1, const Environment &Env1,
                  const RecordStorageLocation &Loc2, const Environment &Env2);

inline bool recordsEqual(const RecordStorageLocation &Loc1,
                         const RecordStorageLocation &Loc2,
                         const Environment &Env) {
  return recordsEqual(Loc1, Env, Loc2, Env);
}

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_RECORDOPS_H
