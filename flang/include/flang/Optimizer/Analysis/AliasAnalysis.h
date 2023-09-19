//===-- AliasAnalysis.h - Alias Analysis in FIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_ANALYSIS_ALIASANALYSIS_H
#define FORTRAN_OPTIMIZER_ANALYSIS_ALIASANALYSIS_H

#include "flang/Common/enum-class.h"
#include "flang/Common/enum-set.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"

namespace fir {

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//
struct AliasAnalysis {
  // Structures to describe the memory source of a value.

  /// Kind of the memory source referenced by a value.
  ENUM_CLASS(SourceKind,
             /// Unique memory allocated by an operation, e.g.
             /// by fir::AllocaOp or fir::AllocMemOp.
             Allocate,
             /// A global object allocated statically on the module level.
             Global,
             /// Memory allocated outside of a function and passed
             /// to the function as a by-ref argument.
             Argument,
             /// Represents memory allocated outside of a function
             /// and passed to the function via host association tuple.
             HostAssoc,
             /// Represents direct memory access whose source cannot be further
             /// determined
             Direct,
             /// Represents memory allocated by unknown means and
             /// with the memory address defined by a memory reading
             /// operation (e.g. fir::LoadOp).
             Indirect,
             /// Starting point to the analysis whereby nothing is known about
             /// the source
             Unknown);

  /// Attributes of the memory source object.
  ENUM_CLASS(Attribute, Target, Pointer, IntentIn);

  struct Source {
    using SourceUnion = llvm::PointerUnion<mlir::SymbolRefAttr, mlir::Value>;
    using Attributes = Fortran::common::EnumSet<Attribute, Attribute_enumSize>;

    /// Source definition of a value.
    SourceUnion u;
    /// Kind of the memory source.
    SourceKind kind;
    /// Value type of the source definition.
    mlir::Type valueType;
    /// Attributes of the memory source object, e.g. Target.
    Attributes attributes;
    /// Have we lost precision following the source such that
    /// even an exact match cannot be MustAlias?
    bool approximateSource;

    /// Print information about the memory source to `os`.
    void print(llvm::raw_ostream &os) const;

    /// Return true, if Target or Pointer attribute is set.
    bool isTargetOrPointer() const;

    /// Return true, if the memory source's `valueType` is a reference type
    /// to an object of derived type that contains a component with POINTER
    /// attribute.
    bool isRecordWithPointerComponent() const;

    /// Return true, if `ty` is a reference type to a boxed
    /// POINTER object or a raw fir::PointerType.
    static bool isPointerReference(mlir::Type ty);
  };

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const AliasAnalysis::Source &op);

  /// Given two values, return their aliasing behavior.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

  /// Return the memory source of a value.
  Source getSource(mlir::Value);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AliasAnalysis::Source &op) {
  op.print(os);
  return os;
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_ALIASANALYSIS_H
