// RUN: %clang -fdepscan-prefix-map=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: %clang -fdepscan-prefix-map==/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: %clang -fdepscan-prefix-map=relative=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: %clang -fdepscan-prefix-map=/=/^bad -### %s 2>&1 | FileCheck %s -check-prefix=INVALID
// INVALID: error: invalid argument '{{.*}}/^bad' to -fdepscan-prefix-map=

// RUN: %clang -fdepscan-prefix-map=/good=/^good -### %s 2>&1 | FileCheck %s
// CHECK: "-fdepscan-prefix-map=/good=/^good"
