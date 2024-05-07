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
//      AFTER:   %[[#SRC:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// AFTER-NEXT:   %[[#SRC_IS_NOT_NULL:]] = cir.cast(ptr_to_bool, %[[#SRC]] : !cir.ptr<!ty_22Base22>), !cir.bool
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#SRC_IS_NOT_NULL]], true {
// AFTER-NEXT:     %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!void>
// AFTER-NEXT:     %[[#BASE_RTTI:]] = cir.const #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#DERIVED_RTTI:]] = cir.const #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#HINT:]] = cir.const #cir.int<0> : !s64i
// AFTER-NEXT:     %[[#RT_CALL_RET:]] = cir.call @__dynamic_cast(%[[#SRC_VOID_PTR]], %[[#BASE_RTTI]], %[[#DERIVED_RTTI]], %[[#HINT]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// AFTER-NEXT:     %[[#CASTED:]] = cir.cast(bitcast, %[[#RT_CALL_RET]] : !cir.ptr<!void>), !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:     cir.yield %[[#CASTED]] : !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     %[[#NULL_PTR:]] = cir.const #cir.ptr<null> : !cir.ptr<!ty_22Derived22>
// AFTER-NEXT:     cir.yield %[[#NULL_PTR]] : !cir.ptr<!ty_22Derived22>
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
// AFTER-NEXT:   %[[#SRC_RTTI:]] = cir.const #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>
// AFTER-NEXT:   %[[#DEST_RTTI:]] = cir.const #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>
// AFTER-NEXT:   %[[#OFFSET_HINT:]] = cir.const #cir.int<0> : !s64i
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

// BEFORE: cir.func @_Z20ptr_cast_to_completeP4Base
// BEFORE:   %{{.+}} = cir.dyn_cast(ptr, %{{.+}} : !cir.ptr<!ty_22Base22>) -> !cir.ptr<!void>
// BEFORE: }

//      AFTER: cir.func @_Z20ptr_cast_to_completeP4Base
//      AFTER:   %[[#SRC:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// AFTER-NEXT:   %[[#SRC_IS_NOT_NULL:]] = cir.cast(ptr_to_bool, %[[#SRC]] : !cir.ptr<!ty_22Base22>), !cir.bool
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#SRC_IS_NOT_NULL]], true {
// AFTER-NEXT:     %[[#VPTR_PTR:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!cir.ptr<!s64i>>
// AFTER-NEXT:     %[[#VPTR:]] = cir.load %[[#VPTR_PTR]] : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// AFTER-NEXT:     %[[#BASE_OFFSET_PTR:]] = cir.vtable.address_point( %[[#VPTR]] : !cir.ptr<!s64i>, vtable_index = 0, address_point_index = -2) : !cir.ptr<!s64i>
// AFTER-NEXT:     %[[#BASE_OFFSET:]] = cir.load %[[#BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// AFTER-NEXT:     %[[#SRC_BYTES_PTR:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#DST_BYTES_PTR:]] = cir.ptr_stride(%[[#SRC_BYTES_PTR]] : !cir.ptr<!u8i>, %[[#BASE_OFFSET]] : !s64i), !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#CASTED_PTR:]] = cir.cast(bitcast, %[[#DST_BYTES_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// AFTER-NEXT:     cir.yield %[[#CASTED_PTR]] : !cir.ptr<!void>
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     %[[#NULL_PTR:]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// AFTER-NEXT:     cir.yield %[[#NULL_PTR]] : !cir.ptr<!void>
// AFTER-NEXT:   }) : (!cir.bool) -> !cir.ptr<!void>
//      AFTER: }
