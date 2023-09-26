// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s

// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s

struct test {
  test();
  ~test();
} t;

int bar() __attribute__((destructor(101)));
int bar2() __attribute__((destructor(65535)));

int bar() {
  return 1;
}

int bar2() {
  return 2;
}

// NO-REGISTER: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I__, ptr null }]
// NO-REGISTER: @llvm.global_dtors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @_Z3barv, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_Z4bar2v, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__D_a, ptr null }]

// REGISTER: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I__, ptr null }, { i32, ptr, ptr } { i32 101, ptr @__GLOBAL_init_101, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__GLOBAL_init_65535, ptr null }]
// REGISTER: @llvm.global_dtors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__D_a, ptr null }, { i32, ptr, ptr } { i32 101, ptr @__GLOBAL_cleanup_101, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__GLOBAL_cleanup_65535, ptr null }]

// REGISTER: define internal void @__GLOBAL_init_101() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(ptr @_Z3barv)
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_init_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(ptr @_Z4bar2v)
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_101() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(ptr @_Z3barv)
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// REGISTER: destruct.call:
// REGISTER:   call void @_Z3barv()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(ptr @_Z4bar2v)
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// REGISTER: destruct.call:
// REGISTER:   call void @_Z4bar2v()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }
