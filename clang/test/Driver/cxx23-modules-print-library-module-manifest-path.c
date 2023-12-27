// Test that -print-library-module-manifest-path finds the correct file.

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%S/Inputs/cxx23_modules \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX %s
// CHECK-LIBCXX: module: ={{.*}}/Inputs/cxx23_modules/usr/lib/x86_64-linux-gnu/modules.json

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/cxx23_modules \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBSTDCXX %s
// CHECK-LIBSTDCXX: module: =
