// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Pure virtual functions are allowed to be defined, but the vtable should still
// point to __cxa_pure_virtual instead of the definition. For destructors, the
// base object destructor (which is not included in the vtable) should be
// defined as usual. The complete object destructors and deleting destructors
// should contain a trap, and the vtable entries for them should point to
// __cxa_pure_virtual.
class C {
  C();
  virtual ~C() = 0;
  virtual void pure() = 0;
};

C::C() = default;
C::~C() = default;
void C::pure() {}

// CHECK: @_ZTV1C = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>
// complete object destructor (D1)
// CHECK-SAME: #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>,
// deleting destructor (D0)
// CHECK-SAME: #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>,
// C::pure
// CHECK-SAME: #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>]>

// The base object destructor should be emitted as normal.
// CHECK-LABEL: cir.func @_ZN1CD2Ev(%arg0: !cir.ptr<!ty_C> loc({{[^)]+}})) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    cir.store %arg0, %0 : !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>
// CHECK-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C>>, !cir.ptr<!ty_C>
// CHECK-NEXT:    cir.return
// CHECK-NEXT:  }

// The complete object destructor should trap.
// CHECK-LABEL: cir.func @_ZN1CD1Ev(%arg0: !cir.ptr<!ty_C> loc({{[^)]+}})) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    cir.store %arg0, %0 : !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>
// CHECK-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C>>, !cir.ptr<!ty_C>
// CHECK-NEXT:    cir.trap
// CHECK-NEXT:  }

// The deleting destructor should trap.
// CHECK-LABEL: cir.func @_ZN1CD0Ev(%arg0: !cir.ptr<!ty_C> loc({{[^)]+}})) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    cir.store %arg0, %0 : !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>
// CHECK-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C>>, !cir.ptr<!ty_C>
// CHECK-NEXT:    cir.trap
// CHECK-NEXT:  }

// C::pure should be emitted as normal.
// CHECK-LABEL: cir.func @_ZN1C4pureEv(%arg0: !cir.ptr<!ty_C> loc({{[^)]+}})) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    cir.store %arg0, %0 : !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>
// CHECK-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C>>, !cir.ptr<!ty_C>
// CHECK-NEXT:    cir.return
// CHECK-NEXT:  }
