/// Tests for SYCL offloading JIT that require Unix commands (rm, ln, mkdir, touch)
// These tests require a fake install tree with a symlinked clang so that D.Dir points to a
// controlled location, allowing us to place a dummy libLLVMSYCL.so where the driver expects it.
// UNSUPPORTED: system-windows, system-cygwin

// Check if path to the SYCL RT is passed to clang-linker-wrapper for SYCL compilation.
// The test also checks if SYCL header include paths are added to the SYCL host and device compilation.

// Check LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON case: library is in lib/<triple>/
// RUN: rm -rf %t && mkdir -p %t/bin %t/lib/x86_64-unknown-linux-gnu
// RUN: touch %t/lib/x86_64-unknown-linux-gnu/libLLVMSYCL.so
// RUN: ln -s %clang %t/bin/clang
// RUN: %t/bin/clang -### -no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK-LSYCL,CHECK-SYCL-HEADERS-HOST,CHECK-SYCL-HEADERS-DEVICE %s
// CHECK-SYCL-HEADERS-DEVICE: "-fsycl-is-device"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"
// CHECK-SYCL-HEADERS-HOST: "-fsycl-is-host"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"
// CHECK-LSYCL: clang-linker-wrapper{{.*}} "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}x86_64-unknown-linux-gnu{{[/\\]+}}libLLVMSYCL.so"

// Check LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF case: library is in lib/ (no triple subdir)
// RUN: rm -rf %t && mkdir -p %t/bin %t/lib
// RUN: touch %t/lib/libLLVMSYCL.so
// RUN: ln -s %clang %t/bin/clang
// RUN: %t/bin/clang -### -no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LSYCL-FLAT %s
// CHECK-LSYCL-FLAT: clang-linker-wrapper{{.*}} "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}libLLVMSYCL.so"
