// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -std=c2x %s -o - | FileCheck %s
// RUN: %clang_cc1 -Wno-error=return-type -triple %itanium_abi_triple -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-CXX

typedef void (*fptrs_t[4])(void);
fptrs_t p __attribute__((noreturn));

void __attribute__((noreturn)) f(void) {
  p[0]();
}
// CHECK: call void
// CHECK-NEXT: unreachable

// CHECK-LABEL: @test_conditional_gnu(
// CHECK:         %cond = select i1 %tobool, ptr @t1, ptr @t2
// CHECK:         call void %cond(
// CHECK:         call void %cond2(
// CHECK-NEXT:    unreachable

// CHECK-CXX-LABEL: @_Z20test_conditional_gnui(
// CHECK-CXX:         %cond{{.*}} = phi ptr [ @_Z2t1i, %{{.*}} ], [ @_Z2t2i, %{{.*}} ]
// CHECK-CXX:         call void %cond{{.*}}(
// CHECK-CXX:         %cond{{.*}} = phi ptr [ @_Z2t1i, %{{.*}} ], [ @_Z2t1i, %{{.*}} ]
// CHECK-CXX:         call void %cond{{.*}}(
// CHECK-CXX-NEXT:    unreachable
void t1(int) __attribute__((noreturn));
void t2(int);
__attribute__((noreturn)) void test_conditional_gnu(int a) {
  // The conditional operator isn't noreturn because t2 isn't.
  (a ? t1 : t2)(a);
  // The conditional operator is noreturn.
  (a ? t1 : t1)(a);
}

// CHECK-LABEL: @test_conditional_Noreturn(
// CHECK:         %cond = select i1 %tobool, ptr @t3, ptr @t2
// CHECK:         call void %cond(
// CHECK:         %cond2 = select i1 %tobool1, ptr @t3, ptr @t3
// CHECK:         call void %cond2(
// CHECK-NEXT:    ret void
_Noreturn void t3(int);
_Noreturn void test_conditional_Noreturn(int a) {
  (a ? t3 : t2)(a);
  (a ? t3 : t3)(a);
}

// CHECK-LABEL: @test_conditional_std(
// CHECK:         %cond = select i1 %tobool, ptr @t4, ptr @t2
// CHECK:         call void %cond(
// CHECK:         %cond2 = select i1 %tobool1, ptr @t4, ptr @t4
// CHECK:         call void %cond2(
// CHECK-NEXT:    ret void
[[noreturn]] void t4(int);
[[noreturn]] void test_conditional_std(int a) {
  (a ? t4 : t2)(a);
  (a ? t4 : t4)(a);
}
