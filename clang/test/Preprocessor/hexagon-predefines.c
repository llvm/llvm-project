// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv5 %s | FileCheck %s -check-prefix CHECK-V5

// CHECK-V5: #define __HEXAGON_ARCH__ 5
// CHECK-V5: #define __HEXAGON_V5__ 1
// CHECK-V5: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv55 %s | FileCheck %s -check-prefix CHECK-V55

// CHECK-V55: #define __HEXAGON_ARCH__ 55
// CHECK-V55: #define __HEXAGON_V55__ 1
// CHECK-V55: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 %s | FileCheck %s -check-prefix CHECK-V60

// CHECK-V60: #define __HEXAGON_ARCH__ 60
// CHECK-V60: #define __HEXAGON_V60__ 1
// CHECK-V60: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 -target-feature +hvx %s | FileCheck %s -check-prefix CHECK-V60HVX

// CHECK-V60HVX: #define __HEXAGON_ARCH__ 60
// CHECK-V60HVX: #define __HEXAGON_V60__ 1
// CHECK-V60HVX: #define __HVX__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 -target-feature +hvx-double  %s | FileCheck %s -check-prefix CHECK-V60HVXD

// CHECK-V60HVXD: #define __HEXAGON_ARCH__ 60
// CHECK-V60HVXD: #define __HEXAGON_V60__ 1
// CHECK-V60HVXD: #define __HVXDBL__ 1
// CHECK-V60HVXD: #define __HVX__ 1
// CHECK-V60HVXD: #define __hexagon__ 1

