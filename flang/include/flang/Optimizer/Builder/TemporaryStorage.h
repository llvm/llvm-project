//===-- Optimizer/Builder/TemporaryStorage.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Utility to create an manipulate vector like temporary storages holding
//  Fortran values or descriptors in HLFIR.
//
//  This is useful to deal with array constructors, and temporary storage
//  inside forall and where constructs where it is not known prior to the
//  construct execution how many values will be stored, or where the values
//  at each iteration may have different shapes or type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_TEMPORARYSTORAGE_H
#define FORTRAN_OPTIMIZER_BUILDER_TEMPORARYSTORAGE_H

#include "flang/Optimizer/HLFIR/HLFIROps.h"

namespace fir {
class FirOpBuilder;
}

namespace hlfir {
class Entity;
}

namespace fir::factory {

/// Structure to create and manipulate counters in generated code.
/// They are used to keep track of the insertion of fetching position in the
/// temporary storages.
/// By default, this counter is implemented with a value in memory and can be
/// incremented inside generated loops or branches.
/// The option canCountThroughLoops can be set to false to get a counter that
/// is a simple SSA value that is swap by its incremented value (hence, the
/// counter cannot count through loops since the SSA value in the loop becomes
/// inaccessible after the loop). This form helps reducing compile times for
/// huge array constructors without implied-do-loops.
struct Counter {
  /// Create a counter set to the initial value.
  Counter(mlir::Location loc, fir::FirOpBuilder &builder,
          mlir::Value initialValue, bool canCountThroughLoops = true);
  /// Return "counter++".
  mlir::Value getAndIncrementIndex(mlir::Location loc,
                                   fir::FirOpBuilder &builder);
  /// Set the counter to the initial value.
  void reset(mlir::Location loc, fir::FirOpBuilder &builder);
  const bool canCountThroughLoops;

private:
  /// Zero for the init/reset.
  mlir::Value initialValue;
  /// One for the increment.
  mlir::Value one;
  /// Index variable or value holding the counter current value.
  mlir::Value index;
};

/// Data structure to stack simple scalars that all have the same type and
/// type parameters, and where the total number of elements that will be pushed
/// is known or can be maximized. It is implemented inlined and does not require
/// runtime.
class HomogeneousScalarStack {
public:
  HomogeneousScalarStack(mlir::Location loc, fir::FirOpBuilder &builder,
                         fir::SequenceType declaredType, mlir::Value extent,
                         llvm::ArrayRef<mlir::Value> lengths,
                         bool allocateOnHeap, bool stackThroughLoops,
                         llvm::StringRef name);

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value);
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder);
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder);
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder);

  /// Move the temporary storage into a rank one array expression value
  /// (hlfir.expr<?xT>). The temporary should not be used anymore after this
  /// call.
  hlfir::Entity moveStackAsArrayExpr(mlir::Location loc,
                                     fir::FirOpBuilder &builder);

private:
  /// Allocate the temporary on the heap.
  const bool allocateOnHeap;
  /// Counter to keep track of the insertion or fetching position.
  Counter counter;
  /// Temporary storage.
  mlir::Value temp;
};
} // namespace fir::factory
#endif // FORTRAN_OPTIMIZER_BUILDER_TEMPORARYSTORAGE_H
