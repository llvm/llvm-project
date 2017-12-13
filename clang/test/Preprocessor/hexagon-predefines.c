// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv5 %s | FileCheck %s -check-prefix CHECK-V5
// CHECK-V5: #define __HEXAGON_ARCH__ 5
// CHECK-V5: #define __HEXAGON_V5__ 1
// CHECK-V5-NOT: #define __HVX_LENGTH__
// CHECK-V5-NOT: #define __HVX__ 1
// CHECK-V5: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv55 %s | FileCheck %s -check-prefix CHECK-V55
// CHECK-V55: #define __HEXAGON_ARCH__ 55
// CHECK-V55: #define __HEXAGON_V55__ 1
// CHECK-V55-NOT: #define __HVX_LENGTH__
// CHECK-V55-NOT: #define __HVX__ 1
// CHECK-V55: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 %s | FileCheck %s -check-prefix CHECK-V60
// CHECK-V60: #define __HEXAGON_ARCH__ 60
// CHECK-V60: #define __HEXAGON_V60__ 1
// CHECK-V60-NOT: #define __HVX_LENGTH__
// CHECK-V60-NOT: #define __HVX__ 1
// CHECK-V60: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv62 %s | FileCheck %s -check-prefix CHECK-V62
// CHECK-V62: #define __HEXAGON_ARCH__ 62
// CHECK-V62: #define __HEXAGON_V62__ 1
// CHECK-V62-NOT: #define __HVX_LENGTH__
// CHECK-V62-NOT: #define __HVX__ 1
// CHECK-V62: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv65 %s | FileCheck %s -check-prefix CHECK-V65
// CHECK-V65: #define __HEXAGON_ARCH__ 65
// CHECK-V65: #define __HEXAGON_V65__ 1
// CHECK-V65-NOT: #define __HVX_LENGTH__
// CHECK-V65-NOT: #define __HVX__ 1
// CHECK-V65: #define __hexagon__ 1

// The HVX flags are explicitly defined by the driver.
// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 \
// RUN: -target-feature +hvxv60 -target-feature +hvx-length64b %s | FileCheck \
// RUN: %s -check-prefix CHECK-V60HVX-64B
// CHECK-V60HVX-64B: #define __HEXAGON_ARCH__ 60
// CHECK-V60HVX-64B: #define __HEXAGON_V60__ 1
// CHECK-V60HVX-64B-NOT: #define __HVXDBL__ 1
// CHECK-V60HVX-64B: #define __HVX_ARCH__ 60
// CHECK-V60HVX-64B: #define __HVX_LENGTH__ 64
// CHECK-V60HVX-64B: #define __HVX__ 1
// CHECK-V60HVX-64B: #define __hexagon__ 1

// RUN: %clang_cc1 -E -dM -triple hexagon-unknown-elf -target-cpu hexagonv60 \
// RUN: -target-feature +hvxv60 -target-feature +hvx-length128b %s | FileCheck \
// RUN: %s -check-prefix CHECK-V60HVX-128B
// CHECK-V60HVX-128B: #define __HEXAGON_ARCH__ 60
// CHECK-V60HVX-128B: #define __HEXAGON_V60__ 1
// CHECK-V60HVX-128B: #define __HVXDBL__ 1
// CHECK-V60HVX-128B: #define __HVX_ARCH__ 60
// CHECK-V60HVX-128B: #define __HVX_LENGTH__ 128
// CHECK-V60HVX-128B: #define __HVX__ 1
// CHECK-V60HVX-128B: #define __hexagon__ 1
