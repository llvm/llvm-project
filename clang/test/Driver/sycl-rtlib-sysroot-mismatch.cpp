// REQUIRES: system-linux && symlinks

// Verify we still generate a path to libLLVMSYCL.so even with sysroot set.

// RUN: rm -rf %t && mkdir -p %t/bin %t/lib
// RUN: touch %t/lib/libLLVMSYCL.so
// RUN: ln -s %clang %t/bin/clang
// RUN: %t/bin/clang -### -no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fsycl --sysroot=/nonexistent-prefix %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FLAT %s

// RUN: rm -rf %t && mkdir -p %t/bin %t/lib/x86_64-unknown-linux-gnu
// RUN: touch %t/lib/x86_64-unknown-linux-gnu/libLLVMSYCL.so
// RUN: ln -s %clang %t/bin/clang
// RUN: %t/bin/clang -### -no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fsycl --sysroot=/nonexistent-prefix %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PER-TARGET %s

// CHECK-FLAT: "{{.*}}/bin/../lib/libLLVMSYCL.so"
// CHECK-PER-TARGET: "{{.*}}/bin/../lib/x86_64-unknown-linux-gnu/libLLVMSYCL.so"
