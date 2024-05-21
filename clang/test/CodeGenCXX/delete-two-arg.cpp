// RUN: %clang_cc1 -triple i686-pc-linux-gnu %s -o - -emit-llvm -verify | FileCheck %s
// expected-no-diagnostics

typedef __typeof(sizeof(int)) size_t;

namespace test1 {
  struct A { void operator delete(void*,size_t); int x; };

  // CHECK-LABEL: define{{.*}} void @_ZN5test11aEPNS_1AE(
  void a(A *x) {
    // CHECK:      load
    // CHECK-NEXT: icmp eq {{.*}}, null
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN5test11AdlEPvj(ptr noundef %{{.*}}, i32 noundef 4)
    delete x;
  }
}

// Check that we make cookies for the two-arg delete even when using
// the global allocator and deallocator.
namespace test2 {
  struct A {
    int x;
    void *operator new[](size_t);
    void operator delete[](void *, size_t);
  };

  // CHECK: define{{.*}} ptr @_ZN5test24testEv()
  A *test() {
    // CHECK:      [[NEW:%.*]] = call noalias noundef nonnull ptr @_Znaj(i32 noundef 44)
    // CHECK-NEXT: store i32 10, ptr [[NEW]]
    // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr [[NEW]], i32 4
    // CHECK-NEXT: ret ptr [[T1]]
    return ::new A[10];
  }

  // CHECK-LABEL: define{{.*}} void @_ZN5test24testEPNS_1AE(
  void test(A *p) {
    // CHECK:      [[P:%.*]] = alloca ptr, align 4
    // CHECK-NEXT: store ptr {{%.*}}, ptr [[P]], align 4
    // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P]], align 4
    // CHECK-NEXT: [[T1:%.*]] = icmp eq ptr [[T0]], null
    // CHECK-NEXT: br i1 [[T1]],
    // CHECK: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i32 -4
    // CHECK-NEXT: [[T5:%.*]] = load i32, ptr [[T3]]
    // CHECK-NEXT: call void @_ZdaPv(ptr noundef [[T3]])
    // CHECK-NEXT: br label
    ::delete[] p;
  }
}

namespace test3 {
  struct A {
    int x;
    void operator delete[](void *, size_t);
  };
  struct B : A {};

  // CHECK-LABEL: define{{.*}} void @_ZN5test34testEv()
  void test() {
    // CHECK:      [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znaj(i32 noundef 24)
    // CHECK-NEXT: store i32 5
    (void) new B[5];
  }
}
