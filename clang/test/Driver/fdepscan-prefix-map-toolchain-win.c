// REQUIRES: system-windows

// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir '' -### %s 2>&1 | FileCheck %s -check-prefix=NONE
// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir relative -### %s 2>&1 | FileCheck %s -check-prefix=NONE
// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir C:\lib\clang\10 -### %s 2>&1 | FileCheck %s -check-prefix=NONE

// NONE-NOT: -fdepscan-prefix-map

// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir C:\tc\10 -### %s 2>&1 | FileCheck %s
// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir C:\tc\lib\clang\10 -### %s 2>&1 | FileCheck %s
// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -resource-dir C:\tc\usr\lib\clang\10 -### %s 2>&1 | FileCheck %s

// CHECK: "-fdepscan-prefix-map" "C:\\tc" "\\^tc"

// Implicit resource-dir
// RUN: %clang -fdepscan-prefix-map-toolchain=\^tc -### %s 2>&1 | FileCheck %s -check-prefix=CHECK_IMPLICIT
// CHECK_IMPLICIT: "-fdepscan-prefix-map" "{{.*}}" "\\^tc"
