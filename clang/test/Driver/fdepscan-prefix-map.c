// REQUIRES: !system-windows

// RUN: not %clang -fdepscan-prefix-map=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang -fdepscan-prefix-map==/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang -fdepscan-prefix-map=relative=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang -fdepscan-prefix-map=/=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// INVALID: error: invalid argument '{{.*}}/^bad' to -fdepscan-prefix-map=

// RUN: %clang -fdepscan-prefix-map=/good=/^good -### %s 2>&1 | FileCheck --check-prefix CHECK_CORRECT %s
// CHECK_CORRECT: "-fdepscan-prefix-map" "/good" "/^good"

// RUN %clang -fdepscan-prefix-map=/a=/^a -fdepscan-prefix-map /b /^b -fdepscan-prefix-map=/c=/^c -fdepscan-prefix-map /d /^d -### %s 2>&1 | FileCheck --check-prefix=CHECK_MIXED %s
// CHECK_MIXED: "-fdepscan-prefix-map" "/a" "/^a" "-fdepscan-prefix-map" "/b" "/^b" "-fdepscan-prefix-map" "/c" "/^c" "-fdepscan-prefix-map" "/d" "/^d"
