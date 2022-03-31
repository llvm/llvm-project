// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -std=c++2a < %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -std=c++2a < %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s

namespace test1 {
  struct Test1 {
    Test1();
    ~Test1();
  } t1, t2;
} // namespace test1

namespace test2 {
  int foo() { return 3; }
  int x = foo();
} // namespace test2

namespace test3 {
  struct Test3 {
    constexpr Test3() {};
    ~Test3() {};
  };

  constinit Test3 t;
} // namespace test3

namespace test4 {
  struct Test4 {
    Test4();
    ~Test4();
  };

  void f() {
    static Test4 staticLocal;
  }
} // namespace test4

// CHECK: @_ZGVZN5test41fEvE11staticLocal = internal global i64 0, align 8
// CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I__, i8* null }]
// CHECK: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_a, i8* null }]

// CHECK: define internal void @__cxx_global_var_init() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1C1Ev(%"struct.test1::Test1"* {{[^,]*}} @_ZN5test12t1E)
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor__ZN5test12t1E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t1E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(%"struct.test1::Test1"* @_ZN5test12t1E)
// CHECK:   ret void
// CHECK: }

// CHECK: declare i32 @atexit(void ()*)

// CHECK: define internal void @__finalize__ZN5test12t1E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test12t1E)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test12t1E()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: declare i32 @unatexit(void ()*)

// CHECK: define internal void @__cxx_global_var_init.1() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1C1Ev(%"struct.test1::Test1"* {{[^,]*}} @_ZN5test12t2E)
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor__ZN5test12t2E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t2E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(%"struct.test1::Test1"* @_ZN5test12t2E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t2E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test12t2E)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test12t2E()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__cxx_global_var_init.2() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK32: %call = call noundef i32 @_ZN5test23fooEv()
// CHECK64: %call = call noundef signext i32 @_ZN5test23fooEv()
// CHECK:   store i32 %call, i32* @_ZN5test21xE
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__cxx_global_var_init.3() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor__ZN5test31tE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test31tE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test35Test3D1Ev(%"struct.test3::Test3"* @_ZN5test31tE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test31tE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test31tE)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test31tE()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define void @_ZN5test41fEv() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = load atomic i8, i8* bitcast (i64* @_ZGVZN5test41fEvE11staticLocal to i8*) acquire, align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK:   %1 = call i32 @__cxa_guard_acquire(i64* @_ZGVZN5test41fEvE11staticLocal)
// CHECK:   %tobool = icmp ne i32 %1, 0
// CHECK:   br i1 %tobool, label %init, label %init.end

// CHECK: init:
// CHECK:   call void @_ZN5test45Test4C1Ev(%"struct.test4::Test4"* {{[^,]*}} @_ZZN5test41fEvE11staticLocal)
// CHECK:   %2 = call i32 @atexit(void ()* @__dtor__ZZN5test41fEvE11staticLocal)
// CHECK:   call void @__cxa_guard_release(i64* @_ZGVZN5test41fEvE11staticLocal)
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZZN5test41fEvE11staticLocal() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test45Test4D1Ev(%"struct.test4::Test4"* @_ZZN5test41fEvE11staticLocal)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZZN5test41fEvE11staticLocal() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZZN5test41fEvE11staticLocal)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZZN5test41fEvE11staticLocal()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__sub_I__() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init()
// CHECK:   call void @__cxx_global_var_init.1()
// CHECK:   call void @__cxx_global_var_init.2()
// CHECK:   call void @__cxx_global_var_init.3()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__D_a() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize__ZZN5test41fEvE11staticLocal()
// CHECK:   call void @__finalize__ZN5test31tE()
// CHECK:   call void @__finalize__ZN5test12t2E()
// CHECK:   call void @__finalize__ZN5test12t1E()
// CHECK:   ret void
// CHECK: }
