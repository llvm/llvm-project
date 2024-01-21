// Test that -print-library-module-manifest-path finds the correct file.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: mkdir -p %t/Inputs/usr/lib/x86_64-linux-gnu
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%t/Inputs \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx-no-module-json.cpp

// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/modules.json
// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%t/Inputs \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx.cpp

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%t/Inputs \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libstdcxx.cpp

//--- libcxx-no-module-json.cpp

// CHECK: <NOT PRESENT>

//--- libcxx.cpp

// CHECK: {{.*}}/Inputs/usr/lib/x86_64-linux-gnu{{/|\\}}modules.json

//--- libstdcxx.cpp

// CHECK: <NOT PRESENT>
