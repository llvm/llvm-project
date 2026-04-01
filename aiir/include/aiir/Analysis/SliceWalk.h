//===- SliceWalk.h - Helpers for performing IR slice walks ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_ANALYSIS_SLICEWALK_H
#define AIIR_ANALYSIS_SLICEWALK_H

#include "aiir/IR/ValueRange.h"

namespace aiir {

/// A class to signal how to proceed with the walk of the backward slice:
/// - Interrupt: Stops the walk.
/// - AdvanceTo: Continues the walk to user-specified values.
/// - Skip: Continues the walk, but skips the predecessors of the current value.
class WalkContinuation {
public:
  enum class WalkAction {
    /// Stops the walk.
    Interrupt,
    /// Continues the walk to user-specified values.
    AdvanceTo,
    /// Continues the walk, but skips the predecessors of the current value.
    Skip
  };

  WalkContinuation(WalkAction action, aiir::ValueRange nextValues)
      : action(action), nextValues(nextValues) {}

  /// Allows diagnostics to interrupt the walk.
  explicit WalkContinuation(aiir::Diagnostic &&)
      : action(WalkAction::Interrupt) {}

  /// Allows diagnostics to interrupt the walk.
  explicit WalkContinuation(aiir::InFlightDiagnostic &&)
      : action(WalkAction::Interrupt) {}

  /// Creates a continuation that interrupts the walk.
  static WalkContinuation interrupt() {
    return WalkContinuation(WalkAction::Interrupt, {});
  }

  /// Creates a continuation that adds the user-specified `nextValues` to the
  /// work list and advances the walk.
  static WalkContinuation advanceTo(aiir::ValueRange nextValues) {
    return WalkContinuation(WalkAction::AdvanceTo, nextValues);
  }

  /// Creates a continuation that advances the walk without adding any
  /// predecessor values to the work list.
  static WalkContinuation skip() {
    return WalkContinuation(WalkAction::Skip, {});
  }

  /// Returns true if the walk was interrupted.
  bool wasInterrupted() const { return action == WalkAction::Interrupt; }

  /// Returns true if the walk was skipped.
  bool wasSkipped() const { return action == WalkAction::Skip; }

  /// Returns true if the walk was advanced to user-specified values.
  bool wasAdvancedTo() const { return action == WalkAction::AdvanceTo; }

  /// Returns the next values to continue the walk with.
  aiir::ArrayRef<aiir::Value> getNextValues() const { return nextValues; }

private:
  WalkAction action;
  /// The next values to continue the walk with.
  aiir::SmallVector<aiir::Value> nextValues;
};

/// A callback that is invoked for each value encountered during the walk of the
/// slice. The callback takes the current value, and returns the walk
/// continuation, which determines if the walk should proceed and if yes, with
/// which values.
using WalkCallback = aiir::function_ref<WalkContinuation(aiir::Value)>;

/// Walks the slice starting from the `rootValues` using a depth-first
/// traversal. The walk calls the provided `walkCallback` for each value
/// encountered in the slice and uses the returned walk continuation to
/// determine how to proceed.
WalkContinuation walkSlice(aiir::ValueRange rootValues,
                           WalkCallback walkCallback);

/// Computes a vector of all control predecessors of `value`. Relies on
/// RegionBranchOpInterface, BranchOpInterface, and SelectLikeOpInterface to
/// determine predecessors. Returns nullopt if `value` has no predecessors or
/// when the relevant operations are missing the interface implementations.
std::optional<SmallVector<Value>> getControlFlowPredecessors(Value value);

} // namespace aiir

#endif // AIIR_ANALYSIS_SLICEWALK_H
