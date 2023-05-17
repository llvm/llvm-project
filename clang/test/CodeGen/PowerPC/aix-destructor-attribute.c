// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck --check-prefix=NO-REGISTER %s

// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit -fregister-global-dtors-with-atexit < %s | \
// RUN:   FileCheck --check-prefix=REGISTER %s

int bar(void) __attribute__((destructor(100)));
int bar2(void) __attribute__((destructor(65535)));
int bar3(int) __attribute__((destructor(65535)));

int bar(void) {
  return 1;
}

int bar2(void) {
  return 2;
}

int bar3(int a) {
  return a;
}

// NO-REGISTER: @llvm.global_dtors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 100, ptr @bar, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @bar2, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @bar3, ptr null }]

// REGISTER: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 100, ptr @__GLOBAL_init_100, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__GLOBAL_init_65535, ptr null }]
// REGISTER: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 100, ptr @__GLOBAL_cleanup_100, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__GLOBAL_cleanup_65535, ptr null }]

// REGISTER: define internal void @__GLOBAL_init_100() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(ptr @bar)
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_init_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @atexit(ptr @bar2)
// REGISTER:   %1 = call i32 @atexit(ptr @bar3)
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_100() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(ptr @bar)
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %destruct.end

// REGISTER: destruct.call:
// REGISTER:   call void @bar()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }

// REGISTER: define internal void @__GLOBAL_cleanup_65535() [[ATTR:#[0-9]+]] {
// REGISTER: entry:
// REGISTER:   %0 = call i32 @unatexit(ptr @bar3)
// REGISTER:   %needs_destruct = icmp eq i32 %0, 0
// REGISTER:   br i1 %needs_destruct, label %destruct.call, label %unatexit.call

// REGISTER: destruct.call:
// REGISTER:   call void @bar3()
// REGISTER:   br label %unatexit.call

// REGISTER: unatexit.call:
// REGISTER:   %1 = call i32 @unatexit(ptr @bar2)
// REGISTER:   %needs_destruct1 = icmp eq i32 %1, 0
// REGISTER:   br i1 %needs_destruct1, label %destruct.call2, label %destruct.end

// REGISTER: destruct.call2:
// REGISTER:   call void @bar2()
// REGISTER:   br label %destruct.end

// REGISTER: destruct.end:
// REGISTER:   ret void
// REGISTER: }
