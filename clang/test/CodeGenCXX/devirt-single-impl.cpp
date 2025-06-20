// Check that speculative devirtualization works without the need for LTO or visibility.
// RUN: %clang_cc1 -fwhole-program-vtables -O1 %s -emit-llvm -o - | FileCheck %s

struct A {
  A(){}
  __attribute__((noinline))
  virtual int virtual1(){return 20;}
  __attribute__((noinline))
  virtual void empty_virtual(){}
};

struct B : A {
  B(){}
  __attribute__((noinline))
  virtual int virtual1() override {return 50;}
  __attribute__((noinline))
  virtual void empty_virtual() override {}
};

// Test that we can apply speculative devirtualization
// without the need for LTO or visibility.
__attribute__((noinline))
int test_devirtual(A *a) {
  // CHECK: %0 = load ptr, ptr %vtable, align 8
  // CHECK-NEXT: %1 = icmp eq ptr %0, @_ZN1B8virtual1Ev
  // CHECK-NEXT: br i1 %1, label %if.true.direct_targ, label %if.false.orig_indirect, !prof !12

  // CHECK: if.true.direct_targ: ; preds = %entry
  // CHECK-NEXT: %2 = tail call noundef i32 @_ZN1B8virtual1Ev(ptr noundef nonnull align 8 dereferenceable(8) %a)
  // CHECK-NEXT: br label %if.end.icp

  // CHECK: if.false.orig_indirect: ; preds = %entry
  // CHECK-NEXT: %call = tail call noundef i32 %0(ptr noundef nonnull align 8 dereferenceable(8) %a)
  // CHECK-NEXT: br label %if.end.icp

  // CHECK: if.end.icp: ; preds = %if.false.orig_indirect, %if.true.direct_targ
  // CHECK-NEXT: %3 = phi i32 [ %call, %if.false.orig_indirect ], [ %2, %if.true.direct_targ ]
  // CHECK-NEXT: ret i32 %3

  return a->virtual1();
}

// Test that we skip devirtualization for empty virtual functions as most probably
// they are used for interfaces.
__attribute__((noinline))
void test_devirtual_empty_fn(A *a) {
  // CHECK: load ptr, ptr %vfn, align 8
  // CHECK-NEXT: tail call void %0(ptr noundef nonnull align 8 dereferenceable(8) %a)
  a->empty_virtual();
}

void test() {
  A *a = new B();
  test_devirtual(a);
  test_devirtual_empty_fn(a);
}