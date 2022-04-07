// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -std=c++2a < %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ \
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
// CHECK: @llvm.global_ctors = appending global [4 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init.1, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init.2, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init.4, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I__, i8* null }]
// CHECK: @llvm.global_dtors = appending global [4 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__finalize__ZN5test12t2E, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__finalize__ZN5test21AIvE8instanceE, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__finalize__ZN5test12t1IiEE, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_a, i8* null }]

// CHECK: define internal void @__cxx_global_var_init() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK32: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* noundef{{[^,]*}} @_ZN5test12t0E, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* noundef{{[^,]*}} @_ZN5test12t0E, i32 noundef signext 2)
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor__ZN5test12t0E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t0E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(%"struct.test1::Test1"* @_ZN5test12t0E)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t0E() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test12t0E)
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
// CHECK:   %0 = load atomic i8, i8* bitcast (i64* @_ZGVN5test12t2E to i8*) acquire, align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK:   %1 = call i32 @__cxa_guard_acquire(i64* @_ZGVN5test12t2E)
// CHECK:   %tobool = icmp ne i32 %1, 0
// CHECK:   br i1 %tobool, label %init, label %init.end

// CHECK: init:
// CHECK32: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* noundef{{[^,]*}} @_ZN5test12t2E, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* noundef{{[^,]*}} @_ZN5test12t2E, i32 noundef signext 2)
// CHECK:   %2 = call i32 @atexit(void ()* @__dtor__ZN5test12t2E)
// CHECK:   call void @__cxa_guard_release(i64* @_ZGVN5test12t2E)
// CHECK:   br label %init.end

// CHECK: init.end:
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
// CHECK:   %0 = load i8, i8* bitcast (i64* @_ZGVN5test21AIvE8instanceE to i8*), align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK:   call void @_ZN5test21AIvEC1Ev(%"struct.test2::A"* {{[^,]*}} @_ZN5test21AIvE8instanceE)
// CHECK:   %1 = call i32 @atexit(void ()* @__dtor__ZN5test21AIvE8instanceE)
// CHECK:   store i8 1, i8* bitcast (i64* @_ZGVN5test21AIvE8instanceE to i8*), align 8
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test21AIvE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test21AIvED1Ev(%"struct.test2::A"* @_ZN5test21AIvE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test21AIvE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test21AIvE8instanceE)
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
// CHECK:   %call = call noundef nonnull align 1 dereferenceable(1) %"struct.test2::A.0"* @_ZN5test23barEv()
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor__ZN5test21AIiE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test21AIiE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test21AIiED1Ev(%"struct.test2::A.0"* @_ZN5test21AIiE8instanceE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test21AIiE8instanceE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test21AIiE8instanceE)
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
// CHECK:   %0 = load i8, i8* bitcast (i64* @_ZGVN5test12t1IiEE to i8*), align 8
// CHECK:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:   br i1 %guard.uninitialized, label %init.check, label %init.end

// CHECK: init.check:
// CHECK32: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* {{[^,]*}} @_ZN5test12t1IiEE, i32 noundef 2)
// CHECK64: call void @_ZN5test15Test1C1Ei(%"struct.test1::Test1"* {{[^,]*}} @_ZN5test12t1IiEE, i32 noundef signext 2)
// CHECK:   %1 = call i32 @atexit(void ()* @__dtor__ZN5test12t1IiEE)
// CHECK:   store i8 1, i8* bitcast (i64* @_ZGVN5test12t1IiEE to i8*), align 8
// CHECK:   br label %init.end

// CHECK: init.end:
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__dtor__ZN5test12t1IiEE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN5test15Test1D1Ev(%"struct.test1::Test1"* @_ZN5test12t1IiEE)
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @__finalize__ZN5test12t1IiEE() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor__ZN5test12t1IiEE)
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
