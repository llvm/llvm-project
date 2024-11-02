// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -std=c++2a < %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -std=c++2a < %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s

namespace test1 {
struct Test1 {
  Test1(int) {}
  ~Test1() {}
};

Test1 t0 = 2;

template <typename T>
Test1 t1 = 2;

inline Test1 t2 = 2;

void foo() {
  (void)&t1<int>;
}
} // namespace test1

namespace test2 {
template <typename = void>
struct A {
  A() {}
  ~A() {}
  static A instance;
};

template <typename T>
A<T> A<T>::instance;
template A<> A<>::instance;

A<int> &bar() {
  A<int> *a = new A<int>;
  return *a;
}
template <>
A<int> A<int>::instance = bar();
} // namespace test2

// CHECK: @_ZGVN5test12t2E = linkonce_odr global i64 0, align 8
// CHECK: @_ZGVN5test21AIvE8instanceE = weak_odr global i64 0, align 8
// CHECK: @_ZGVN5test12t1IiEE = linkonce_odr global i64 0, align 8
// CHECK: @llvm.global_ctors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.2, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.4, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I__, ptr null }]
// CHECK: @llvm.global_dtors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__finalize__ZN5test12t2E, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__finalize__ZN5test21AIvE8instanceE, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__finalize__ZN5test12t1IiEE, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__D_a, ptr null }]

// CHECK: define internal void @__cxx_global_var_init() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK32: call void @_ZN5test15Test1C1Ei(ptr noundef{{[^,]*}} @_ZN5test12t0E, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(ptr noundef{{[^,]*}} @_ZN5test12t0E, i32 noundef signext 2)
// CHECK:   %0 = call i32 @atexit(ptr @__dtor__ZN5test12t0E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t0E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(ptr @_ZN5test12t0E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t0E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(ptr @__dtor__ZN5test12t0E)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test12t0E()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__cxx_global_var_init.1() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = load atomic i8, ptr @_ZGVN5test12t2E acquire, align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK:   %1 = call i32 @__cxa_guard_acquire(ptr @_ZGVN5test12t2E)
// CHECK:   %tobool = icmp ne i32 %1, 0
// CHECK:   br i1 %tobool, label %init, label %init.end

// CHECK: init:
// CHECK32: call void @_ZN5test15Test1C1Ei(ptr noundef{{[^,]*}} @_ZN5test12t2E, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(ptr noundef{{[^,]*}} @_ZN5test12t2E, i32 noundef signext 2)
// CHECK:   %2 = call i32 @atexit(ptr @__dtor__ZN5test12t2E)
// CHECK:   call void @__cxa_guard_release(ptr @_ZGVN5test12t2E)
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t2E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(ptr @_ZN5test12t2E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t2E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(ptr @__dtor__ZN5test12t2E)
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
// CHECK:   %0 = load i8, ptr @_ZGVN5test21AIvE8instanceE, align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK:   call void @_ZN5test21AIvEC1Ev(ptr {{[^,]*}} @_ZN5test21AIvE8instanceE)
// CHECK:   %1 = call i32 @atexit(ptr @__dtor__ZN5test21AIvE8instanceE)
// CHECK:   store i8 1, ptr @_ZGVN5test21AIvE8instanceE, align 8
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test21AIvE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test21AIvED1Ev(ptr @_ZN5test21AIvE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test21AIvE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(ptr @__dtor__ZN5test21AIvE8instanceE)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test21AIvE8instanceE()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__cxx_global_var_init.3() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_ZN5test23barEv()
// CHECK:   %0 = call i32 @atexit(ptr @__dtor__ZN5test21AIiE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test21AIiE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test21AIiED1Ev(ptr @_ZN5test21AIiE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test21AIiE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(ptr @__dtor__ZN5test21AIiE8instanceE)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test21AIiE8instanceE()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__cxx_global_var_init.4() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = load i8, ptr @_ZGVN5test12t1IiEE, align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK32: call void @_ZN5test15Test1C1Ei(ptr {{[^,]*}} @_ZN5test12t1IiEE, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(ptr {{[^,]*}} @_ZN5test12t1IiEE, i32 noundef signext 2)
// CHECK:   %1 = call i32 @atexit(ptr @__dtor__ZN5test12t1IiEE)
// CHECK:   store i8 1, ptr @_ZGVN5test12t1IiEE, align 8
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t1IiEE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(ptr @_ZN5test12t1IiEE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t1IiEE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(ptr @__dtor__ZN5test12t1IiEE)
// CHECK:   %needs_destruct = icmp eq i32 %0, 0
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// CHECK: destruct.call:
// CHECK:   call void @__dtor__ZN5test12t1IiEE()
// CHECK:   br label %destruct.end

// CHECK: destruct.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__sub_I__() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init()
// CHECK:   call void @__cxx_global_var_init.3()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__D_a() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize__ZN5test21AIiE8instanceE()
// CHECK:   call void @__finalize__ZN5test12t0E()
// CHECK:   ret void
// CHECK: }
