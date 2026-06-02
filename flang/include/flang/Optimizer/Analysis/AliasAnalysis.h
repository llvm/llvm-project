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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

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
  ENUM_CLASS(Attribute, Target, Pointer, IntentIn, CrayPointer, CrayPointee);

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
  //  flang -fc1 -emit-fir test.f90 -o test.fir
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

      /// A value definition denoting the place where the corresponding
      /// source variable was instantiated by the front-end.
      /// Currently, it is the result of [hl]fir.declare of the source,
      /// if we can reach it.
      /// It helps to identify the scope where the corresponding variable
      /// was defined in the original Fortran source, e.g. when MLIR
      /// inlining happens an inlined fir.declare of the callee's
      /// dummy argument identifies the scope where the source
      /// may be treated as a dummy argument.
      mlir::Operation *instantiationPoint;

      /// Whether the source was reached following data or box reference
      bool isData{false};
    };

    /// Represents a step in the access path from a root variable to the
    /// memory location being queried. Built during the backward walk in
    /// getSource() and stored in forward (root-to-leaf) order.
    struct PathStep {
      enum class Kind {
        /// Named component access, e.g. x%field.
        Component,
        /// Loading a POINTER box (fir.load of !fir.box<!fir.ptr<...>>).
        /// The resulting address depends on pointer association at runtime.
        PointerDeref,
        /// Loading an ALLOCATABLE box (fir.load of !fir.box<!fir.heap<...>>).
        AllocDeref,
      };
      Kind kind;
      /// For Component steps: the field name from hlfir.designate's component
      /// attribute or fir.coordinate_of's field_indices (mapped through the
      /// record type). Null for non-Component steps.
      mlir::StringAttr component;

      bool operator==(const PathStep &o) const {
        return kind == o.kind && component == o.component;
      }
      bool operator!=(const PathStep &o) const { return !(*this == o); }
    };

    /// The access path from the root variable to the queried memory location.
    /// For example, given:
    ///   type(outer) :: x   ! outer has component "in" of type inner
    ///                      ! inner has pointer component "p"
    /// the target address of x%in%p (i.e. the data x%in%p points to) is:
    ///   [{Component,"in"}, {Component,"p"}, {PointerDeref}]
    /// where PointerDeref represents loading the pointer descriptor and
    /// extracting the target address (fir.load + fir.box_addr in the IR).
    /// This enables disambiguation of accesses to different components of
    /// the same derived-type variable and tracks pointer dereferences for
    /// pointer/target aliasing (Fortran 2018 8.5.7, 15.5.2.13).
    struct AccessPath {
      llvm::SmallVector<PathStep, 4> steps;

      /// Whether the path is approximate (e.g. contains array indexing or
      /// went through PackArrayOp). An approximate path cannot yield
      /// MustAlias.
      bool isApproximate{false};

      /// Return true if any step is a PointerDeref.
      bool hasPointerDeref() const {
        return llvm::any_of(steps, [](const PathStep &s) {
          return s.kind == PathStep::Kind::PointerDeref;
        });
      }

      bool operator==(const AccessPath &o) const {
        return isApproximate == o.isApproximate && steps == o.steps;
      }
      bool operator!=(const AccessPath &o) const { return !(*this == o); }

      void print(llvm::raw_ostream &os) const;
    };

    /// A snapshot taken when getSource() walks through an [hl]fir.declare.
    /// Records the Fortran procedure scope of the declare (the dummy_scope
    /// SSA value -- nullptr for non-dummy frames), the declare's result SSA
    /// value, and the access path and attributes accumulated FROM THE LEAF
    /// UP TO (and including) this declare. alias() uses these snapshots to
    /// rebuild intermediate Sources rooted at shared-scope declares via
    /// buildSourceAtDeclare().
    ///
    /// Relationship to the enclosing Source's top-level fields:
    ///   - Source::accessPath / Source::attributes / Source::approximateSource
    ///     describe the walk's TERMINAL stop ("root-of-walk to leaf"). They
    ///     are what every existing alias rule and external consumer reads.
    ///   - The root-closest ScopedOrigin is a snapshot at the last
    ///     declare crossed before that terminal stop. Because getSource()
    ///     walks backwards (leaf -> root) and push_back()s each snapshot,
    ///     this is the LAST element pushed, i.e. scopedOrigins.back() --
    ///     see the ordering note on the scopedOrigins member below (front
    ///     = leaf-closest, back = root-closest). Its accessPath is a
    ///     prefix-truncation of Source::accessPath, and its attributes are
    ///     a subset of Source::attributes; in the common case where the
    ///     terminal stop IS a declare (dummy with dummy_scope, host-assoc,
    ///     OpenMP private), they coincide.
    /// We keep both because:
    ///   1. alias(Source, Source, ...) and external consumers (AddAliasTags,
    ///      print, downstream analyses) read the top-level fields.
    ///   2. When the terminal stop is not a declare (e.g. fir.alloca,
    ///      fir.address_of with TARGET-attributed global), the top-level
    ///      captures attributes the root-closest snapshot does not.
    ///   3. When the walk terminates at Unknown without crossing any
    ///      declare, scopedOrigins is empty and the top-level is the only
    ///      answer.
    struct ScopedOrigin {
      /// The dummy_scope SSA value governing the declare. May be null Value
      /// when the declare has no explicit dummy_scope and no fir.dummy_scope
      /// op dominates it (e.g. globals at module scope).
      mlir::Value scope;
      /// Result SSA value of the [hl]fir.declare op.
      mlir::Value declValue;
      /// Path from the declare (treated as root) to the leaf Value the
      /// original getSource() call was started from, in root-to-leaf order.
      AccessPath accessPath;
      /// Attributes accumulated from the leaf up to and including this
      /// declare (includes getAttrsFromVariable(declare) and any
      /// path-acquired bits such as Pointer from intermediate box loads).
      Attributes attributes;
      /// Whether the path is approximate at the moment of the snapshot.
      bool approximateSource{false};
      /// Whether the walk was following data (vs. a box reference) at the
      /// moment of the snapshot.
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
    /// The structured access path from the root variable.
    AccessPath accessPath;
    /// Source object is used in an internal procedure via host association.
    bool isCapturedInInternalProcedure{false};
    /// Per-declare checkpoints collected as getSource() walked through
    /// [hl]fir.declare operations, ordered from leaf-closest (front) to
    /// root-closest (back). Empty when no declare was crossed (e.g. the
    /// walk terminated at Unknown).
    llvm::SmallVector<ScopedOrigin, 4> scopedOrigins;

    /// Print information about the memory source to `os`.
    void print(llvm::raw_ostream &os) const;

    /// Return true, if Target or Pointer attribute is set.
    bool isTargetOrPointer() const;

    /// Return true, if Target attribute is set.
    bool isTarget() const;

    /// Return true, if Pointer attribute is set.
    bool isPointer() const;

    /// Return true, if CrayPointer attribute is set.
    bool isCrayPointer() const;

    /// Return true, if CrayPointee attribute is set.
    bool isCrayPointee() const;

    /// Return true, if CrayPointer or CrayPointee attribute is set.
    bool isCrayPointerOrPointee() const;

    bool isDummyArgument() const;
    bool isData() const;
    bool isBoxData() const;

    /// Is this source a variable from the Fortran source?
    bool isFortranUserVariable() const;

    /// @name Dummy Argument Aliasing
    ///
    /// Check conditions related to dummy argument aliasing.
    ///
    /// For all uses, a result of false can prevent MayAlias from being
    /// reported, so the list of cases where false is returned is conservative.

    ///@{
    /// The address of a (possibly host associated) dummy argument of the
    /// current function?
    bool mayBeDummyArgOrHostAssoc() const;
    /// \c mayBeDummyArgOrHostAssoc and the address of a pointer?
    bool mayBePtrDummyArgOrHostAssoc() const;
    /// The address of an actual argument of the current function?
    bool mayBeActualArg() const;
    /// \c mayBeActualArg and the address of either a pointer or a composite
    /// with a pointer component?
    bool mayBeActualArgWithPtr(const mlir::Value *val) const;
    ///@}

    mlir::Type getType() const;
  };

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const AliasAnalysis::Source &op);

  /// Given the values and their sources, return their aliasing behavior.
  mlir::AliasResult alias(Source lhsSrc, Source rhsSrc, mlir::Value lhs,
                          mlir::Value rhs);

  /// Given two values, return their aliasing behavior.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

  /// Return the modify-reference behavior of operations inside `region` on
  /// `location`. Contrary to getModRef(operation, location), this will visit
  /// nested regions recursively according to the HasRecursiveMemoryEffects
  /// trait.
  mlir::ModRefResult getModRef(mlir::Region &region, mlir::Value location);

  /// Return the memory source of a value.
  /// If getLastInstantiationPoint is true, the search for the source
  /// will stop at [hl]fir.declare if it represents a dummy
  /// argument declaration (i.e. it has the dummy_scope operand).
  /// If collectScopedOrigins is false, the per-declare ScopedOrigin
  /// snapshots are not collected (used internally by buildSourceAtDeclare
  /// to reuse getSource purely for declare classification without the
  /// bookkeeping side effect).
  fir::AliasAnalysis::Source getSource(mlir::Value,
                                       bool getLastInstantiationPoint = false,
                                       bool collectScopedOrigins = true);

  /// Return true, if `ty` is a reference type to a boxed
  /// POINTER object or a raw fir::PointerType.
  static bool isPointerReference(mlir::Type ty);

private:
  /// Build an intermediate Source rooted at the declare captured by the
  /// snapshot. Reuses getSource(declValue) for the SourceKind / origin
  /// classification (with collectScopedOrigins=false), then overrides
  /// accessPath/attributes/approximateSource/origin.isData from the
  /// snapshot so the returned Source represents "declare-as-root,
  /// original-query-as-leaf".
  Source buildSourceAtDeclare(const Source::ScopedOrigin &so);

  /// Return the dummy_scope SSA value governing \p declareOp.
  /// Prefers the declare's explicit getDummyScope() operand; otherwise
  /// falls back to the result of the dominating fir.dummy_scope op in
  /// the parent func. Returns a null Value when no scope is found
  /// (e.g. globals at module scope).
  mlir::Value getDeclarationScope(mlir::Operation *declareOp);

  /// Return true, if `ty` is a reference type to an object of derived type
  /// that contains a component with POINTER attribute.
  static bool isRecordWithPointerComponent(mlir::Type ty);

  /// Return the symbol table nearest to the given operation.
  /// If a SymbolTable has not been cached in symTabMap,
  /// it will be created, which may be expensive.
  const mlir::SymbolTable *getNearestSymbolTable(mlir::Operation *from);

  /// Return true if the given symbol may correspond to a Fortran variable
  /// with a TARGET attribute. 'from' is used to find the nearest
  /// SymbolTable (by calling getNearestSymbolTable()).
  bool symbolMayHaveTargetAttr(mlir::SymbolRefAttr symbol,
                               mlir::Operation *from);

  /// Return true if the given operation is a call to a Fortran user
  /// procedure.
  bool isCallToFortranUserProcedure(mlir::Operation *op);

  /// Returns the modify-reference behavior of the given call
  /// operation `op` on `var`. If `op` is not a fir.call, then
  /// it returns the conservative ModAndRef result.
  mlir::ModRefResult getCallModRef(mlir::Operation *op, mlir::Value var);

  /// A map between operations with OpTrait::SymbolTable
  /// and the SymbolTable objects associated with them.
  /// TODO: it might be better to initialize just a single SymbolTable
  /// during fir::AliasAnalysis construction, e.g. by giving
  /// the constructor the operation from which the nearest SymbolTable
  /// should be looked up. This implies that the users will have to
  /// specify proper operation (e.g. 'module') so that the discovered
  /// SymbolTable contains all the symbols that may appear during
  /// the aliasing queries through the constructed AliasAnalysis
  /// entity. On ther other hand, this approach may be too expensive
  /// for the clients that create AliasAnalysis on the fly for just
  /// a few values that are likely not globals.
  /// We can have both modes for different clients.
  llvm::DenseMap<mlir::Operation *, mlir::SymbolTable> symTabMap;

  /// Per-function caches used by getDeclarationScope() to map a
  /// fir.declare without an explicit dummy_scope operand to its
  /// dominating fir.dummy_scope op. Lazily populated. Mirrors the
  /// logic in flang/lib/Optimizer/Transforms/AddAliasTags.cpp::
  /// PassState::processFunctionScopes / getDeclarationScope. The cache
  /// stores fir.dummy_scope ops as mlir::Operation * pointers to avoid
  /// pulling FIROps.h into this header; the .cpp casts back as needed.
  ///
  /// TODO: this duplicates the scope-mapping logic in AddAliasTags.cpp.
  /// AddAliasTags should reuse AliasAnalysis::getDeclarationScope (and
  /// the ScopedOrigin snapshots collected by getSource) instead of
  /// maintaining its own PassState::processFunctionScopes, so the two
  /// places cannot diverge.
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<mlir::DominanceInfo>>
      domInfoCache;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 16>>
      sortedScopeCache;
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
