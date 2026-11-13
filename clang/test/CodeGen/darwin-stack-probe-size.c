// REQUIRES: aarch64-registered-target

// Check that the Darwin 16K default stack probe threshold is present in the IR

// (1) arm64 Darwin targets get 16k threshold
// RUN: %clang -target arm64-apple-macosx15.0 %s -S -emit-llvm -o - | FileCheck %s --check-prefix=ARM64
// RUN: %clang -target arm64-apple-ios17.0 %s -S -emit-llvm -o - | FileCheck %s --check-prefix=ARM64
// ARM64: define{{.*}} @large_frame() #0
// ARM64: attributes #0 = { {{.*}}"probe-stack"="__chkstk_darwin" "stack-probe-size"="16384"
// ARM64: !{i32 8, !"stack-probe-size", i32 16384}

// (2) manually specified -mstack-probe-size=4096 gets the LLVM default of 4k threshold
// The 4k threshold is spelled as the *absence* of the attribute and the module
// flag, so the CHECK-NOT must be anchored to bound its scan region: matching
// "probe-stack" first starts it inside the attribute list.
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=4096 -S -emit-llvm -o - | FileCheck %s --check-prefix=MANUAL-4K
// MANUAL-4K: attributes #0 = { {{.*}}"probe-stack"="__chkstk_darwin"
// MANUAL-4K-NOT: "stack-probe-size"

// (3) any other manually specified -mstack-probe-size
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=8192 -S -emit-llvm -o - | FileCheck %s --check-prefix=MANUAL-8K
// MANUAL-8K: attributes #0 = { {{.*}}"probe-stack"="__chkstk_darwin" "stack-probe-size"="8192"
// MANUAL-8K: !{i32 8, !"stack-probe-size", i32 8192}

// (4) kernel/kext code gets 16k threshold
// RUN: %clang -target arm64-apple-macosx15.0 %s -mkernel -S -emit-llvm -o - | FileCheck %s --check-prefix=KERNEL
// RUN: %clang -target arm64-apple-macosx15.0 %s -fapple-kext -S -emit-llvm -o - | FileCheck %s --check-prefix=KERNEL
// KERNEL: attributes #0 = { {{.*}}"probe-stack"="__chkstk_darwin" "stack-probe-size"="16384"
// KERNEL: !{i32 8, !"stack-probe-size", i32 16384}

// (5) -ffreestanding disables Darwin stack checking
// RUN: %clang -target arm64-apple-macosx15.0 %s -ffreestanding -S -emit-llvm -o - | FileCheck %s --check-prefix=FREESTANDING
// FREESTANDING: define{{.*}} @large_frame() #0
// FREESTANDING-NOT: "probe-stack"
// FREESTANDING-NOT: "stack-probe-size"

// (6) -ffreestanding + -fstack-check keeps 4k threshold (LLVM default)
// RUN: %clang -target arm64-apple-macosx15.0 %s -ffreestanding -fstack-check -S -emit-llvm -o - | FileCheck %s --check-prefix=FREESTANDING-STACK-CHECK
// FREESTANDING-STACK-CHECK: attributes #0 = { {{.*}}"probe-stack"="__chkstk_darwin"
// FREESTANDING-STACK-CHECK-NOT: "stack-probe-size"

void large_frame(void) {
  volatile char buf[8192];
  buf[0] = 1;
}
