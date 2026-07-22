// REQUIRES: system-linux && symlinks

// Verify we still generate a path to libLLVMSYCL.so even with sysroot set.

// RUN: rm -rf %t && mkdir -p %t/bin %t/lib
// RUN: touch %t/lib/libLLVMSYCL.so
// RUN: ln -s %clang %t/bin/clang
// RUN: %t/bin/clang -### -no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fsycl --sysroot=/nonexistent-prefix %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "{{.*}}/bin/../lib/libLLVMSYCL.so"
