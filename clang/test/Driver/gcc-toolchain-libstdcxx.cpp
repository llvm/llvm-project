// UNSUPPORTED: system-windows

// This file tests that the GCC installation directory detection takes
// the libstdc++ includes into account.  In each directory
// Inputs/gcc_toolchain_libstdcxx/gccX, the installation directory for
// gcc X should be picked in the future since it is the directory with
// the largest version number which also contains the libstdc++
// include directory.

// RUN: %clang --target=x86_64-linux-gnu --gcc-toolchain=%S/Inputs/gcc_toolchain_libstdcxx/gcc10/usr -v 2>&1 | FileCheck %s --check-prefix=GCC10
// GCC10: clang: warning: future releases of the clang compiler will prefer GCC installations containing libstdc++ include directories; '[[PREFIX:.*gcc_toolchain_libstdcxx/gcc10/usr/lib/gcc/x86_64-linux-gnu]]/10' would be chosen over '[[PREFIX]]/12' [-Wgcc-install-dir-libstdcxx]
// GCC10: Found candidate GCC installation: [[PREFIX]]/10
// GCC10: Found candidate GCC installation: [[PREFIX]]/11
// GCC10: Found candidate GCC installation: [[PREFIX]]/12
// GCC10: Selected GCC installation: [[PREFIX]]/12

// RUN: %clang --target=x86_64-linux-gnu --gcc-toolchain=%S/Inputs/gcc_toolchain_libstdcxx/gcc11/usr -v 2>&1 | FileCheck %s --check-prefix=ONLY_GCC11
// ONLY_GCC11: clang: warning: future releases of the clang compiler will prefer GCC installations containing libstdc++ include directories; '[[PREFIX:.*gcc_toolchain_libstdcxx/gcc11/usr/lib/gcc/x86_64-linux-gnu]]/11' would be chosen over '[[PREFIX]]/12' [-Wgcc-install-dir-libstdcxx]
// ONLY_GCC11: Found candidate GCC installation: [[PREFIX]]/10
// ONLY_GCC11: Found candidate GCC installation: [[PREFIX]]/11
// ONLY_GCC11: Found candidate GCC installation: [[PREFIX]]/12
// ONLY_GCC11: Selected GCC installation: [[PREFIX]]/12

// RUN: %clang --target=x86_64-linux-gnu --gcc-toolchain=%S/Inputs/gcc_toolchain_libstdcxx/gcc12/usr -v 2>&1 | FileCheck %s --check-prefix=GCC12
// GCC12: Found candidate GCC installation: [[PREFIX:.*gcc_toolchain_libstdcxx/gcc12/usr/lib/gcc/x86_64-linux-gnu]]/10
// GCC12: Found candidate GCC installation: [[PREFIX]]/11
// GCC12: Found candidate GCC installation: [[PREFIX]]/12
// GCC12: Selected GCC installation: [[PREFIX]]/12
