// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Check that pure and deleted virtual functions are correctly emitted in the
// vtable.
class A {
  A();
  virtual void pure() = 0;
  virtual void deleted() = delete;
};

A::A() = default;

// CHECK: @_ZTV1A = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_deleted_virtual> : !cir.ptr<!u8i>]>
// CHECK: cir.func private @__cxa_pure_virtual()
// CHECK: cir.func private @__cxa_deleted_virtual()
