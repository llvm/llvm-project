// RUN: %clang_cc1 -triple i686-pc-linux-gnu %s -o - -emit-llvm -verify | FileCheck %s
// expected-no-diagnostics

typedef __typeof(sizeof(int)) size_t;

namespace test1 {
  struct A { void operator delete(void*,size_t); int x; };

  // CHECK-LABEL: define void @_ZN5test11aEPNS_1AE(
  void a(A *x) {
    // CHECK:      load
    // CHECK-NEXT: icmp eq {{.*}}, null
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN5test11AdlEPvj(i8* %{{.*}}, i32 4)
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

  // CHECK: define [[A:%.*]]* @_ZN5test24testEv()
  A *test() {
    // CHECK:      [[NEW:%.*]] = call i8* @_Znaj(i32 44)
    // CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[NEW]] to i32*
    // CHECK-NEXT: store i32 10, i32* [[T0]]
    // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[NEW]], i32 4
    // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[A]]*
    // CHECK-NEXT: ret [[A]]* [[T2]]
    return ::new A[10];
  }

  // CHECK-LABEL: define void @_ZN5test24testEPNS_1AE(
  void test(A *p) {
    // CHECK:      [[P:%.*]] = alloca [[A]]*, align 4
    // CHECK-NEXT: store [[A]]* {{%.*}}, [[A]]** [[P]], align 4
    // CHECK-NEXT: [[T0:%.*]] = load [[A]]*, [[A]]** [[P]], align 4
    // CHECK-NEXT: [[T1:%.*]] = icmp eq [[A]]* [[T0]], null
    // CHECK-NEXT: br i1 [[T1]],
    // CHECK:      [[T2:%.*]] = bitcast [[A]]* [[T0]] to i8*
    // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i32 -4
    // CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i32*
    // CHECK-NEXT: [[T5:%.*]] = load i32, i32* [[T4]]
    // CHECK-NEXT: call void @_ZdaPv(i8* [[T3]])
    // CHECK-NEXT: br label
    ::delete[] p;
  }
}

// rdar://problem/8913519
namespace test3 {
  struct A {
    int x;
    void operator delete[](void *, size_t);
  };  
  struct B : A {};

  // CHECK-LABEL: define void @_ZN5test34testEv()
  void test() {
    // CHECK:      call i8* @_Znaj(i32 24)
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: store i32 5
    (void) new B[5];
  }
}
