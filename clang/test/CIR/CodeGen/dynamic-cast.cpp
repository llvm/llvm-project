// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// BEFORE: #dyn_cast_info__ZTI4Base__ZTI7Derived = #cir.dyn_cast_info<#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>, @__dynamic_cast, @__cxa_bad_cast, #cir.int<0> : !s64i>
// BEFORE: !ty_22Base22 = !cir.struct
// BEFORE: !ty_22Derived22 = !cir.struct

Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}

// BEFORE: cir.func @_Z8ptr_castP4Base
// BEFORE:   %{{.+}} = cir.dyn_cast(ptr, %{{.+}} : !cir.ptr<!ty_22Base22>, #dyn_cast_info__ZTI4Base__ZTI7Derived) -> !cir.ptr<!ty_22Derived22>
// BEFORE: }

//      AFTER: cir.func @_Z8ptr_castP4Base
//      AFTER:   %[[#SRC_IS_NULL:]] = cir.cast(ptr_to_bool, %{{.+}} : !cir.ptr<!ty_22Base22>), !cir.bool
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#SRC_IS_NULL]], true {
// AFTER-NEXT:     %[[#NULL:]] = cir.const(#cir.ptr<null> : !cir.ptr<!ty_22Derived22>) : !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:     cir.yield %[[#NULL]] : !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %2 : !cir.ptr<!ty_22Base22>), !cir.ptr<!void>
// AFTER-NEXT:     %[[#SRC_RTTI:]] = cir.const(#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#DEST_RTTI:]] = cir.const(#cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#OFFSET_HINT:]] = cir.const(#cir.int<0> : !s64i) : !s64i
// AFTER-NEXT:     %[[#CASTED_PTR:]] = cir.call @__dynamic_cast(%[[#SRC_VOID_PTR]], %[[#SRC_RTTI]], %[[#DEST_RTTI]], %[[#OFFSET_HINT]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// AFTER-NEXT:     %[[#RESULT:]] = cir.cast(bitcast, %[[#CASTED_PTR]] : !cir.ptr<!void>), !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:     cir.yield %[[#RESULT]] : !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:   }) : (!cir.bool) -> !cir.ptr<!ty_22Derived22>
//      AFTER: }

Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}

// BEFORE: cir.func @_Z8ref_castR4Base
// BEFORE:   %{{.+}} = cir.dyn_cast(ref, %{{.+}} : !cir.ptr<!ty_22Base22>, #dyn_cast_info__ZTI4Base__ZTI7Derived) -> !cir.ptr<!ty_22Derived22>
// BEFORE: }

//      AFTER: cir.func @_Z8ref_castR4Base
//      AFTER:   %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %{{.+}} : !cir.ptr<!ty_22Base22>), !cir.ptr<!void>
// AFTER-NEXT:   %[[#SRC_RTTI:]] = cir.const(#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// AFTER-NEXT:   %[[#DEST_RTTI:]] = cir.const(#cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// AFTER-NEXT:   %[[#OFFSET_HINT:]] = cir.const(#cir.int<0> : !s64i) : !s64i
// AFTER-NEXT:   %[[#CASTED_PTR:]] = cir.call @__dynamic_cast(%[[#SRC_VOID_PTR]], %[[#SRC_RTTI]], %[[#DEST_RTTI]], %[[#OFFSET_HINT]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// AFTER-NEXT:   %[[#CASTED_PTR_IS_NOT_NULL:]] = cir.cast(ptr_to_bool, %[[#CASTED_PTR]] : !cir.ptr<!void>), !cir.bool
// AFTER-NEXT:   %[[#CASTED_PTR_IS_NULL:]] = cir.unary(not, %[[#CASTED_PTR_IS_NOT_NULL]]) : !cir.bool, !cir.bool
// AFTER-NEXT:   cir.if %[[#CASTED_PTR_IS_NULL]] {
// AFTER-NEXT:     cir.call @__cxa_bad_cast() : () -> ()
// AFTER-NEXT:     cir.unreachable
// AFTER-NEXT:   }
// AFTER-NEXT:   %{{.+}} = cir.cast(bitcast, %[[#CASTED_PTR]] : !cir.ptr<!void>), !cir.ptr<!ty_22Derived22>
//      AFTER: }

void *ptr_cast_to_complete(Base *ptr) {
  return dynamic_cast<void *>(ptr);
}

//      BEFORE: cir.func @_Z20ptr_cast_to_completeP4Base
//      BEFORE:   %[[#V19:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// BEFORE-NEXT:   %[[#V20:]] = cir.cast(ptr_to_bool, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.bool
// BEFORE-NEXT:   %[[#V21:]] = cir.unary(not, %[[#V20]]) : !cir.bool, !cir.bool
// BEFORE-NEXT:   %{{.+}} = cir.ternary(%[[#V21]], true {
// BEFORE-NEXT:     %[[#V22:]] = cir.const(#cir.ptr<null> : !cir.ptr<!void>) : !cir.ptr<!void>
// BEFORE-NEXT:     cir.yield %[[#V22]] : !cir.ptr<!void>
// BEFORE-NEXT:   }, false {
// BEFORE-NEXT:     %[[#V23:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!cir.ptr<!s64i>>
// BEFORE-NEXT:     %[[#V24:]] = cir.load %[[#V23]] : cir.ptr <!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// BEFORE-NEXT:     %[[#V25:]] = cir.vtable.address_point( %[[#V24]] : !cir.ptr<!s64i>, vtable_index = 0, address_point_index = -2) : cir.ptr <!s64i>
// BEFORE-NEXT:     %[[#V26:]] = cir.load %[[#V25]] : cir.ptr <!s64i>, !s64i
// BEFORE-NEXT:     %[[#V27:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!u8i>
// BEFORE-NEXT:     %[[#V28:]] = cir.ptr_stride(%[[#V27]] : !cir.ptr<!u8i>, %[[#V26]] : !s64i), !cir.ptr<!u8i>
// BEFORE-NEXT:     %[[#V29:]] = cir.cast(bitcast, %[[#V28]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// BEFORE-NEXT:     cir.yield %[[#V29]] : !cir.ptr<!void>
// BEFORE-NEXT:   }) : (!cir.bool) -> !cir.ptr<!void>
