// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s

struct test {
  test();
  ~test();
} t;

int bar() __attribute__((destructor(100)));
int bar2() __attribute__((destructor(65535)));
int bar3(int) __attribute__((destructor(65535)));

int bar() {
  return 1;
}

int bar2() {
  return 2;
}

int bar3(int a) {
  return a;
}

// NO-REGISTER: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I__, i8* null }]
// NO-REGISTER: @llvm.global_dtors = appending global [4 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 100, void ()* bitcast (i32 ()* @_Z3barv to void ()*), i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* bitcast (i32 ()* @_Z4bar2v to void ()*), i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* bitcast (i32 (i32)* @_Z4bar3i to void ()*), i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_a, i8* null }]

// REGISTER: @llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I__, i8* null }, { i32, void ()*, i8* } { i32 100, void ()* @__GLOBAL_init_100, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__GLOBAL_init_65535, i8* null }]
// REGISTER: @llvm.global_dtors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_a, i8* null }, { i32, void ()*, i8* } { i32 100, void ()* @__GLOBAL_cleanup_100, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__GLOBAL_cleanup_65535, i8* null }]

// REGISTER: define internal void @__GLOBAL_init_100() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(void ()* bitcast (i32 ()* @_Z3barv to void ()*))
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_init_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(void ()* bitcast (i32 ()* @_Z4bar2v to void ()*))
// REGISTER:   %1 = call i32 @atexit(void ()* bitcast (i32 (i32)* @_Z4bar3i to void ()*))
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_100() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(void ()* bitcast (i32 ()* @_Z3barv to void ()*))
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// REGISTER: destruct.call:
// REGISTER:   call void bitcast (i32 ()* @_Z3barv to void ()*)()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(void ()* bitcast (i32 (i32)* @_Z4bar3i to void ()*))
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %unatexit.call

// REGISTER: destruct.call:
// REGISTER:   call void bitcast (i32 (i32)* @_Z4bar3i to void ()*)()
// REGISTER:   br label %unatexit.call

// REGISTER: unatexit.call:
// REGISTER:   %1 = call i32 @unatexit(void ()* bitcast (i32 ()* @_Z4bar2v to void ()*))
// REGISTER:   %needs_destruct1 = icmp eq i32 %1, 0
// REGISTER:   br i1 %needs_destruct1, label %destruct.call2, label %destruct.end

// REGISTER: destruct.call2:
// REGISTER:   call void bitcast (i32 ()* @_Z4bar2v to void ()*)()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }
