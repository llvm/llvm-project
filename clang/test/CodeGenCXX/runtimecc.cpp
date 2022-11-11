// RUN: %clang_cc1 %s -triple=armv7-apple-darwin10 -emit-llvm -o - -fexceptions -fcxx-exceptions | FileCheck %s

// Check that we annotate all compiler-synthesized runtime calls and
// functions with the actual ABI-determined CC.  This usually doesn't
// matter as long as we're internally consistent (and the LLVM-default
// CC is consistent with the real one), but it's possible for user
// translation units to define these runtime functions (or, equivalently,
// for us to get LTO'ed with such a translation unit), and then the
// mismatch will kill us.
//
// rdar://12818655

// CHECK: [[A:%.*]] = type { double }

namespace test0 {
  struct A {
    double d;
    A();
    ~A();
  };

  A global;
// CHECK-LABEL:    define internal void @__cxx_global_var_init()
// CHECK:      call noundef ptr @_ZN5test01AC1Ev(ptr {{[^,]*}} @_ZN5test06globalE)
// CHECK-NEXT: call i32 @__cxa_atexit(ptr @_ZN5test01AD1Ev, ptr @_ZN5test06globalE, ptr @__dso_handle) [[NOUNWIND:#[0-9]+]]
// CHECK-NEXT: ret void
}

// CHECK: declare i32 @__cxa_atexit(ptr, ptr, ptr) [[NOUNWIND]]

namespace test1 {
  void test() {
    throw 0;
  }

// CHECK-LABEL:    define{{.*}} void @_ZN5test14testEv()
// CHECK:      [[T0:%.*]] = call ptr @__cxa_allocate_exception(i32 4) [[NOUNWIND]]
// CHECK-NEXT: store i32 0, ptr [[T0]]
// CHECK-NEXT: call void @__cxa_throw(ptr [[T0]], ptr @_ZTIi, ptr null) [[NORETURN:#[0-9]+]]
// CHECK-NEXT: unreachable
}

// CHECK: declare ptr @__cxa_allocate_exception(i32)

// CHECK: declare void @__cxa_throw(ptr, ptr, ptr)

// CHECK-LABEL: define internal void @_GLOBAL__sub_I_runtimecc.cpp()
// CHECK:   call void @__cxx_global_var_init()


// CHECK: attributes [[NOUNWIND]] = { nounwind }
// CHECK: attributes [[NORETURN]] = { noreturn }
