//===- CIRPointsToAnalysis.h - Points-to analysis for CIR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PointsToAnalysis, which computes the set of objects each
// value in an operation may point to.
//
// The analysis is inclusion based and flow insensitive: a pointer stored into
// a memory location at any point may be loaded back from it at any point.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace cir {

/// The objects a value may point to.
///
/// An object is identified by the value that allocates it, which for now is
/// always the result of a cir.alloca. A set may also contain "unknown", which
/// stands for any object the analysis does not track: memory reached through a
/// function argument or a global, memory written by code the analysis cannot
/// see, and any tracked object whose address has escaped.
class PointsToSet {
public:
  /// The tracked objects in the set.
  llvm::ArrayRef<mlir::Value> getObjects() const {
    return objects.getArrayRef();
  }

  /// Whether the set contains "unknown". "unknown" may stand for any object
  /// whose address escaped, including tracked ones, so a set that contains it
  /// may point to any object.
  bool containsUnknown() const { return unknown; }

  /// Whether `object` is in the set.
  bool contains(mlir::Value object) const { return objects.contains(object); }

  /// Whether the set has no tracked object and no "unknown".
  bool empty() const { return objects.empty() && !unknown; }

  /// Whether the set grew past the size limit of the analysis and was replaced
  /// by "unknown". Such a set no longer names the objects it held.
  bool hitSizeLimit() const { return sizeLimitHit; }

  /// Print the set as the sorted names of the cir.alloca operations of its
  /// objects, followed by "unknown" if the set contains it, e.g. "[a, b]".
  void print(llvm::raw_ostream &os) const;

private:
  friend class PointsToAnalysis;

  /// Add `object` to the set. Return true if the set changed.
  bool insert(mlir::Value object) { return objects.insert(object); }

  /// Add "unknown" to the set. Return true if the set changed.
  bool insertUnknown() { return !std::exchange(unknown, true); }

  /// Add every member of `other` to the set. Return true if the set changed.
  bool merge(const PointsToSet &other);

  /// Remove the tracked objects and keep only "unknown". The set stays this way
  /// from then on.
  void replaceWithUnknown();

  llvm::SmallSetVector<mlir::Value, 4> objects;
  bool unknown = false;
  bool sizeLimitHit = false;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const PointsToSet &set) {
  set.print(os);
  return os;
}

/// Limits on the work done by PointsToAnalysis. When the operation is too
/// large, or the sets take too long to compute, the analysis gives up, and
/// every value points to "unknown".
struct PointsToLimits {
  /// The largest number of operations the analysis is computed for.
  unsigned maxOperations = 5000;

  /// The work allowed to compute the sets, counted as the number of times an
  /// operation is visited. It is `workPerOperation` for each operation, and at
  /// least `minWork`.
  unsigned workPerOperation = 32;
  unsigned minWork = 100000;

  /// The largest number of objects a set holds. A set that grows past it is
  /// replaced by "unknown", and its objects escape.
  unsigned maxSetSize = 100;
};

/// Computes the points-to set of every value in an operation.
///
/// The analysis looks at the operations nested in the operation it is built
/// for and tracks the addresses of cir.alloca objects through cir.store and
/// cir.load. Any other use of a tracked address is treated as an escape: the
/// object may then be read or written by code the analysis does not model, so
/// its memory is assumed to hold "unknown", and everything stored in it escapes
/// too. Values the analysis does not compute a set for, such as function
/// arguments or the results of operations it does not model, point to
/// "unknown". A null pointer constant points to no object.
///
/// The sets describe the IR at the time the analysis is built and are not
/// updated when it changes.
///
/// When optimization remarks are enabled for the "points-to" category, the
/// analysis reports the set of every loaded pointer as an analysis remark, and
/// what made it lose precision as missed remarks: the objects that escaped,
/// the sets that were replaced by "unknown", and the limits it hit.
class PointsToAnalysis {
public:
  explicit PointsToAnalysis(mlir::Operation *op,
                            const PointsToLimits &limits = PointsToLimits());
  PointsToAnalysis(const PointsToAnalysis &) = delete;
  PointsToAnalysis &operator=(const PointsToAnalysis &) = delete;

  /// Return the objects `value` may point to. Values that were not analyzed
  /// point to "unknown".
  const PointsToSet &getPointsTo(mlir::Value value) const;

  /// Whether the analysis hit one of its operation or work limits, in which
  /// case every value points to "unknown".
  bool hitLimit() const { return limitHit; }

private:
  /// Compute the sets of the values in `op`.
  void compute(mlir::Operation *op);

  /// Apply the effect of `op` on the points-to sets. Return true if any set
  /// changed.
  bool visit(mlir::Operation *op);

  /// Add the members of `from` to `to`, replacing `to` by "unknown" if it grows
  /// past the size limit. `op` is the operation being visited. Return true if
  /// anything changed.
  bool addTo(PointsToSet &to, const PointsToSet &from, mlir::Operation *op);

  /// Record that `objects` may be accessed by code the analysis cannot see,
  /// because of `op`. Return true if anything changed.
  bool markEscaped(llvm::ArrayRef<mlir::Value> objects, mlir::Operation *op);

  /// Drop every set, so that every value points to "unknown", and record that
  /// a limit was hit.
  void dropAllSets();

  /// Emit the remarks described above for the operations nested in `root`.
  void emitRemarks(mlir::Operation *root) const;

  PointsToLimits limits;

  /// The objects each value may point to.
  llvm::DenseMap<mlir::Value, PointsToSet> valuePointsTo;

  /// The objects the memory of each tracked object may point to.
  llvm::DenseMap<mlir::Value, PointsToSet> memoryPointsTo;

  /// Tracked objects whose address escaped, and the operation that first made
  /// each of them escape.
  llvm::MapVector<mlir::Value, mlir::Operation *> escaped;

  /// The operations that made a set grow past the size limit, and the number of
  /// objects the set held.
  llvm::SmallVector<std::pair<mlir::Operation *, size_t>> sizeLimitHits;

  /// The set returned for values that were not analyzed.
  PointsToSet unknownSet;

  bool limitHit = false;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOANALYSIS_H
