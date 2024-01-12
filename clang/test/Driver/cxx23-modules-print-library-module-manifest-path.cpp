// Test that -print-library-module-manifest-path finds the correct file.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%S/Inputs/cxx23_modules \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx.cpp

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/cxx23_modules \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck -- libstdcxx.cpp

//--- libcxx.cpp

// The final path separator differs on Windows and Linux.
// CHECK: {{.*}}/Inputs/cxx23_modules/usr/lib/x86_64-linux-gnu{{[\/]}}.modules.json

//--- libstdcxx.cpp

// CHECK: <NOT PRESENT>
