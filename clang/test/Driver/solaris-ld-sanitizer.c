/// General tests that the ld -z relax=transtls workaround is only applied
/// on Solaris/amd64. Note that we use sysroot to make these tests
/// independent of the host system.

/// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang --target=sparc-sun-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang -fsanitize=undefined --target=sparc-sun-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check sparc-sun-solaris2.11, 64bit
// RUN: %clang -m64 --target=sparc-sun-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check sparc-sun-solaris2.11, 64bit
// RUN: %clang -m64 -fsanitize=undefined --target=sparc-sun-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check i386-pc-solaris2.11, 32bit
// RUN: %clang --target=i386-pc-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check i386-pc-solaris2.11, 32bit
// RUN: %clang -fsanitize=undefined --target=i386-pc-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

/// Check i386-pc-solaris2.11, 64bit
// RUN: %clang -m64 --target=i386-pc-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s

// CHECK-LD-NOT: "-z" "relax=transtls"

/// Check i386-pc-solaris2.11, 64bit
// RUN: %clang -m64 -fsanitize=undefined --target=i386-pc-solaris2.11 %s -### -fuse-ld= \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X64-UBSAN %s
// RUN: %clang -m64 -fsanitize=undefined --target=i386-pc-solaris2.11 %s -### -fuse-ld=gld \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-GLD-X64-UBSAN %s

// CHECK-LD-X64-UBSAN: "-z" "relax=transtls"
// CHECK-GLD-X64-UBSAN-NOT: "-z" "relax=transtls"

/// General tests that the ld -z now workaround is only applied on
/// Solaris/i386 with shared libclang_rt.asan. Note that we use sysroot to
/// make these tests independent of the host system.

/// Check i386-pc-solaris2.11, 32bit, shared libclang_rt.asan
// RUN: %clang -fsanitize=address -shared-libasan --target=i386-pc-solaris2.11 %s -### 2>&1 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32-ASAN-SHARED %s
// CHECK-LD-X32-ASAN-SHARED: "-z" "now"

/// Check i386-pc-solaris2.11, 32bit, static libclang_rt.asan
// RUN: %clang -fsanitize=address --target=i386-pc-solaris2.11 %s -### 2>&1 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32-ASAN %s
// CHECK-LD-X32-ASAN-NOT: "-z" "now"
