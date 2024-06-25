// Test that -print-library-module-manifest-path finds the correct file.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: mkdir -p %t/Inputs/usr/lib/x86_64-linux-gnu
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.a

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx-no-module-json.cpp

// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.modules.json
// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx.cpp

// RUN: rm %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.a
// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libc++ \
// RUN:     -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libcxx-no-shared-lib.cpp

// RUN: %clang -print-library-module-manifest-path \
// RUN:     -stdlib=libstdc++ \
// RUN:     -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:     --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck libstdcxx.cpp

//--- libcxx-no-module-json.cpp

// CHECK: <NOT PRESENT>

//--- libcxx.cpp

// CHECK: {{.*}}/Inputs/usr/lib/x86_64-linux-gnu{{/|\\}}libc++.modules.json

//--- libcxx-no-shared-lib.cpp

// Note this might find a different path depending whether search path
// contains a different libc++.so.
// CHECK: {{.*}}libc++.modules.json

//--- libstdcxx.cpp

// CHECK: <NOT PRESENT>
