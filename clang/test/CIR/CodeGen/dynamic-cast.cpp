// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Base {
  virtual ~Base();
};
// CHECK: !ty_22Base22 = !cir.struct

struct Derived : Base {};
// CHECK: !ty_22Derived22 = !cir.struct

// CHECK: cir.func private @__dynamic_cast(!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>

Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}
//      CHECK: cir.func @_Z8ptr_castP4Base
//      CHECK:   %[[#V1:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// CHECK-NEXT:   %[[#V2:]] = cir.cast(ptr_to_bool, %[[#V1]] : !cir.ptr<!ty_22Base22>), !cir.bool
// CHECK-NEXT:   %[[#V3:]] = cir.unary(not, %[[#V2]]) : !cir.bool, !cir.bool
// CHECK-NEXT:   %{{.+}} = cir.ternary(%[[#V3]], true {
// CHECK-NEXT:     %[[#V4:]] = cir.const(#cir.ptr<null> : !cir.ptr<!ty_22Derived22>) : !cir.ptr<!ty_22Derived22>
// CHECK-NEXT:     cir.yield %[[#V4]] : !cir.ptr<!ty_22Derived22>
// CHECK-NEXT:   }, false {
// CHECK-NEXT:     %[[#V5:]] = cir.const(#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V6:]] = cir.const(#cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V7:]] = cir.const(#cir.int<0> : !s64i) : !s64i
// CHECK-NEXT:     %[[#V8:]] = cir.cast(bitcast, %2 : !cir.ptr<!ty_22Base22>), !cir.ptr<!void>
// CHECK-NEXT:     %[[#V9:]] = cir.call @__dynamic_cast(%[[#V8]], %[[#V5]], %[[#V6]], %[[#V7]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// CHECK-NEXT:     %[[#V10:]] = cir.cast(bitcast, %[[#V9]] : !cir.ptr<!void>), !cir.ptr<!ty_22Derived22>
// CHECK-NEXT:     cir.yield %[[#V10]] : !cir.ptr<!ty_22Derived22>
// CHECK-NEXT:   }) : (!cir.bool) -> !cir.ptr<!ty_22Derived22>

// CHECK: cir.func private @__cxa_bad_cast()

Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}

//      CHECK: cir.func @_Z8ref_castR4Base
//      CHECK:   %[[#V11:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// CHECK-NEXT:   %[[#V12:]] = cir.const(#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// CHECK-NEXT:   %[[#V13:]] = cir.const(#cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
// CHECK-NEXT:   %[[#V14:]] = cir.const(#cir.int<0> : !s64i) : !s64i
// CHECK-NEXT:   %[[#V15:]] = cir.cast(bitcast, %[[#V11]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!void>
// CHECK-NEXT:   %[[#V16:]] = cir.call @__dynamic_cast(%[[#V15]], %[[#V12]], %[[#V13]], %[[#V14]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// CHECK-NEXT:   %[[#V17:]] = cir.cast(ptr_to_bool, %[[#V16]] : !cir.ptr<!void>), !cir.bool
// CHECK-NEXT:   %[[#V18:]] = cir.unary(not, %[[#V17]]) : !cir.bool, !cir.bool
// CHECK-NEXT:   cir.if %[[#V18]] {
// CHECK-NEXT:     cir.call @__cxa_bad_cast() : () -> ()
// CHECK-NEXT:     cir.unreachable
// CHECK-NEXT:   }
// CHECK-NEXT:   %{{.+}} = cir.cast(bitcast, %[[#V16]] : !cir.ptr<!void>), !cir.ptr<!ty_22Derived22>

void *ptr_cast_to_complete(Base *ptr) {
  return dynamic_cast<void *>(ptr);
}

//      CHECK: cir.func @_Z20ptr_cast_to_completeP4Base
//      CHECK:   %[[#V19:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// CHECK-NEXT:   %[[#V20:]] = cir.cast(ptr_to_bool, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.bool
// CHECK-NEXT:   %[[#V21:]] = cir.unary(not, %[[#V20]]) : !cir.bool, !cir.bool
// CHECK-NEXT:   %{{.+}} = cir.ternary(%[[#V21]], true {
// CHECK-NEXT:     %[[#V22:]] = cir.const(#cir.ptr<null> : !cir.ptr<!void>) : !cir.ptr<!void>
// CHECK-NEXT:     cir.yield %[[#V22]] : !cir.ptr<!void>
// CHECK-NEXT:   }, false {
// CHECK-NEXT:     %[[#V23:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!cir.ptr<!s64i>>
// CHECK-NEXT:     %[[#V24:]] = cir.load %[[#V23]] : cir.ptr <!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CHECK-NEXT:     %[[#V25:]] = cir.vtable.address_point( %[[#V24]] : !cir.ptr<!s64i>, vtable_index = 0, address_point_index = -2) : cir.ptr <!s64i>
// CHECK-NEXT:     %[[#V26:]] = cir.load %[[#V25]] : cir.ptr <!s64i>, !s64i
// CHECK-NEXT:     %[[#V27:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V28:]] = cir.ptr_stride(%[[#V27]] : !cir.ptr<!u8i>, %[[#V26]] : !s64i), !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V29:]] = cir.cast(bitcast, %[[#V28]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// CHECK-NEXT:     cir.yield %[[#V29]] : !cir.ptr<!void>
// CHECK-NEXT:   }) : (!cir.bool) -> !cir.ptr<!void>
