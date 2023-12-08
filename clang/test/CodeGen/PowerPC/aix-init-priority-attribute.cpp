// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm < %s | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm < %s | \
// RUN:   FileCheck %s

struct test {
  test() {}
  ~test() {}
};

__attribute__((init_priority(200)))
test t1;
__attribute__((init_priority(200)))
test t2;
__attribute__((init_priority(300)))
test t3;
__attribute__((init_priority(150)))
test t4;
test t5;

// CHECK: @llvm.global_ctors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 150, ptr @_GLOBAL__I_000150, ptr null }, { i32, ptr, ptr } { i32 200, ptr @_GLOBAL__I_000200, ptr null }, { i32, ptr, ptr } { i32 300, ptr @_GLOBAL__I_000300, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I__, ptr null }]
// CHECK: @llvm.global_dtors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 150, ptr @_GLOBAL__a_000150, ptr null }, { i32, ptr, ptr } { i32 200, ptr @_GLOBAL__a_000200, ptr null }, { i32, ptr, ptr } { i32 300, ptr @_GLOBAL__a_000300, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__D_a, ptr null }]

// CHECK: define internal void @_GLOBAL__I_000150() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init.3()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__I_000200() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init()
// CHECK:   call void @__cxx_global_var_init.1()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__I_000300() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init.2()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__sub_I__() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init.4()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__a_000150() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize_t4()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__a_000200() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize_t2()
// CHECK:   call void @__finalize_t1()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__a_000300() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize_t3()
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__D_a() [[ATTR:#[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize_t5()
// CHECK:   ret void
// CHECK: }
