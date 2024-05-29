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
             /// Represents memory allocated by unknown means and
             /// with the memory address defined by a memory reading
             /// operation (e.g. fir::LoadOp).
             Indirect,
             /// Starting point to the analysis whereby nothing is known about
             /// the source
             Unknown);

  /// Attributes of the memory source object.
  ENUM_CLASS(Attribute, Target, Pointer, IntentIn);

  // See
  // https://discourse.llvm.org/t/rfc-distinguish-between-data-and-non-data-in-fir-alias-analysis/78759/1
  //
  // It is possible, while following the source of a memory reference through
  // the use-def chain, to arrive at the same origin, even though the starting
  // points were known to not alias.
  //
  // clang-format off
  // Example:
  //  ------------------- test.f90 --------------------
  //  module top
  //    real, pointer :: a(:)
  //  end module
  //  
  //  subroutine test()
  //    use top
  //    a(1) = 1
  //  end subroutine
  //  -------------------------------------------------
  // 
  //  flang-new -fc1 -emit-fir test.f90 -o test.fir
  //
  //  ------------------- test.fir --------------------
  //  fir.global @_QMtopEa : !fir.box<!fir.ptr<!fir.array<?xf32>>> 
  //  
  //  func.func @_QPtest() {
  //    %c1 = arith.constant 1 : index
  //    %cst = arith.constant 1.000000e+00 : f32
  //    %0 = fir.address_of(@_QMtopEa) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  //    %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtopEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  //    %2 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  //    ...
  //    %5 = fir.array_coor %2 %c1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
  //    fir.store %cst to %5 : !fir.ref<f32>
  //    return
  //  }
  //  -------------------------------------------------
  //
  // With high level operations, such as fir.array_coor, it is possible to
  // reach into the data wrapped by the box (the descriptor). Therefore when
  // asking about the memory source of %5, we are really asking about the
  // source of the data of box %2.
  //
  // When asking about the source of %0 which is the address of the box, we
  // reach the same source as in the first case: the global @_QMtopEa. Yet one
  // source refers to the data while the other refers to the address of the box
  // itself.
  //
  // To distinguish between the two, the isData flag has been added, whereby
  // data is defined as any memory reference that is not a box reference.
  // Additionally, because it is relied on in HLFIR lowering, we allow querying
  // on a box SSA value, which is interpreted as querying on its data.
  //
  // So in the above example, !fir.ref<f32> and !fir.box<!fir.ptr<!fir.array<?xf32>>> is data, 
  // while !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> is not data.

  // This also applies to function arguments. In the example below, %arg0
  // is data, %arg1 is not data but a load of %arg1 is.
  // 
  // func.func @_QFPtest2(%arg0: !fir.ref<f32>, %arg1: !fir.ref<!fir.box<!fir.ptr<f32>>> )  {
  //    %0 = fir.load %arg1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  //    ... }
  //
  // clang-format on

  struct Source {
    using SourceUnion = llvm::PointerUnion<mlir::SymbolRefAttr, mlir::Value>;
    using Attributes = Fortran::common::EnumSet<Attribute, Attribute_enumSize>;

    struct SourceOrigin {
      /// Source definition of a value.
      SourceUnion u;

      /// Whether the source was reached following data or box reference
      bool isData{false};
    };

    SourceOrigin origin;

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

    bool isDummyArgument() const;
    bool isData() const;
    bool isBoxData() const;

    mlir::Type getType() const;

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

inline bool operator==(const AliasAnalysis::Source::SourceOrigin &lhs,
                       const AliasAnalysis::Source::SourceOrigin &rhs) {
  return lhs.u == rhs.u && lhs.isData == rhs.isData;
}
inline bool operator!=(const AliasAnalysis::Source::SourceOrigin &lhs,
                       const AliasAnalysis::Source::SourceOrigin &rhs) {
  return !(lhs == rhs);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AliasAnalysis::Source &op) {
  op.print(os);
  return os;
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_ALIASANALYSIS_H
