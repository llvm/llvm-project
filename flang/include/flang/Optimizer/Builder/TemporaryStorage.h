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

  ///  "fetch" cannot be called right after "pushValue" because the counter is
  ///  both used for pushing and fetching.
  bool canBeFetchedAfterPush() const { return false; }

private:
  /// Allocate the temporary on the heap.
  const bool allocateOnHeap;
  /// Counter to keep track of the insertion or fetching position.
  Counter counter;
  /// Temporary storage.
  mlir::Value temp;
};

/// Structure to hold the value of a single entity.
class SimpleCopy {
public:
  SimpleCopy(mlir::Location loc, fir::FirOpBuilder &builder,
             hlfir::Entity source, llvm::StringRef tempName);

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value) {
    assert(false && "must not be called: value already set");
  }
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder){};
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder) {
    return copy.getBase();
  }
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder);
  bool canBeFetchedAfterPush() const { return true; }

public:
  /// Temporary storage for the copy.
  hlfir::AssociateOp copy;
};

/// Structure to keep track of a simple mlir::Value. This is useful
/// when a value does not need an in memory copy because it is
/// already saved in an SSA value that will be accessible at the fetching
/// point.
class SSARegister {
public:
  SSARegister(){};

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value) {
    ssaRegister = value;
  }
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder){};
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder) {
    return ssaRegister;
  }
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder) {}
  bool canBeFetchedAfterPush() const { return true; }

public:
  /// Temporary storage for the copy.
  mlir::Value ssaRegister;
};

/// Data structure to stack any kind of values with the same static type and
/// rank. Each value may have different type parameters, bounds, and dynamic
/// type. Fetching value N will return a value with the same dynamic type,
/// bounds, and type parameters as the Nth value that was pushed. It is
/// implemented using runtime.
class AnyValueStack {
public:
  AnyValueStack(mlir::Location loc, fir::FirOpBuilder &builder,
                mlir::Type valueStaticType);

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value);
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder);
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder);
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder);
  bool canBeFetchedAfterPush() const { return true; }

private:
  /// Keep the original value type. Values may be stored by the runtime
  /// with a different type (i1 cannot be passed by descriptor).
  mlir::Type valueStaticType;
  /// Runtime cookie created by the runtime. It is a pointer to an opaque
  /// runtime data structure that manages the stack.
  mlir::Value opaquePtr;
  /// Counter to keep track of the fetching position.
  Counter counter;
  /// Allocatable box passed to the runtime when fetching the values.
  mlir::Value retValueBox;
};

/// Data structure to stack any kind of variables with the same static type and
/// rank. Each variable may have different type parameters, bounds, and dynamic
/// type. Fetching variable N will return a variable with the same address,
/// dynamic type, bounds, and type parameters as the Nth variable that was
/// pushed. It is implemented using runtime.
class AnyVariableStack {
public:
  AnyVariableStack(mlir::Location loc, fir::FirOpBuilder &builder,
                   mlir::Type valueStaticType);

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value);
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder);
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder);
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder);
  bool canBeFetchedAfterPush() const { return true; }

private:
  /// Keep the original variable type.
  mlir::Type variableStaticType;
  /// Runtime cookie created by the runtime. It is a pointer to an opaque
  /// runtime data structure that manages the stack.
  mlir::Value opaquePtr;
  /// Counter to keep track of the fetching position.
  Counter counter;
  /// Pointer box passed to the runtime when fetching the values.
  mlir::Value retValueBox;
};

class TemporaryStorage;

/// Data structure to stack vector subscripted entity shape and
/// element addresses. AnyVariableStack allows saving vector subscripted
/// entities element addresses, but when saving several vector subscripted
/// entities on a stack, and if the context does not allow retrieving the
/// vector subscript entities shapes, these shapes must be saved too.
class AnyVectorSubscriptStack : public AnyVariableStack {
public:
  AnyVectorSubscriptStack(mlir::Location loc, fir::FirOpBuilder &builder,
                          mlir::Type valueStaticType,
                          bool shapeCanBeSavedAsRegister, int rank);
  void pushShape(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value shape);
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder);
  mlir::Value fetchShape(mlir::Location loc, fir::FirOpBuilder &builder);
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder);
  bool canBeFetchedAfterPush() const { return true; }

private:
  std::unique_ptr<TemporaryStorage> shapeTemp;
  // If the shape is saved inside a descriptor (as extents),
  // keep track of the descriptor type.
  std::optional<mlir::Type> boxType;
};

/// Generic wrapper over the different sorts of temporary storages.
class TemporaryStorage {
public:
  template <typename T>
  TemporaryStorage(T &&impl) : impl{std::forward<T>(impl)} {}

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value value) {
    std::visit([&](auto &temp) { temp.pushValue(loc, builder, value); }, impl);
  }
  void resetFetchPosition(mlir::Location loc, fir::FirOpBuilder &builder) {
    std::visit([&](auto &temp) { temp.resetFetchPosition(loc, builder); },
               impl);
  }
  mlir::Value fetch(mlir::Location loc, fir::FirOpBuilder &builder) {
    return std::visit([&](auto &temp) { return temp.fetch(loc, builder); },
                      impl);
  }
  void destroy(mlir::Location loc, fir::FirOpBuilder &builder) {
    std::visit([&](auto &temp) { temp.destroy(loc, builder); }, impl);
  }
  /// Can "fetch" be called to get the last value pushed with
  /// "pushValue"?
  bool canBeFetchedAfterPush() const {
    return std::visit([&](auto &temp) { return temp.canBeFetchedAfterPush(); },
                      impl);
  }

  template <typename T>
  T &cast() {
    return std::get<T>(impl);
  }

private:
  std::variant<HomogeneousScalarStack, SimpleCopy, SSARegister, AnyValueStack,
               AnyVariableStack, AnyVectorSubscriptStack>
      impl;
};
} // namespace fir::factory
#endif // FORTRAN_OPTIMIZER_BUILDER_TEMPORARYSTORAGE_H
