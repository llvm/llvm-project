// RUN: %clang -target arm-eabi -x c -E -dM %s -o - | FileCheck %s
// RUN: %clang -target thumb-eabi -x c -E -dM %s -o - | FileCheck %s

// CHECK-NOT: __ARM_64BIT_STATE
// CHECK-NOT: __ARM_ARCH_ISA_A64
// CHECK-NOT: __ARM_BIG_ENDIAN
// CHECK:     __ARM_32BIT_STATE 1
// CHECK:     __ARM_ACLE 200

// RUN: %clang -target armeb-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-BIGENDIAN
// RUN: %clang -target thumbeb-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-BIGENDIAN

// CHECK-BIGENDIAN: __ARM_BIG_ENDIAN 1

// RUN: %clang -target armv7-none-linux-eabi -mno-unaligned-access -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-UNALIGNED

// CHECK-UNALIGNED-NOT: __ARM_FEATURE_UNALIGNED

// RUN: %clang -target arm-none-linux-eabi -march=armv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V4

// CHECK-V4-NOT: __ARM_ARCH_ISA_THUMB
// CHECK-V4-NOT: __ARM_ARCH_PROFILE
// CHECK-V4-NOT: __ARM_FEATURE_CLZ
// CHECK-V4-NOT: __ARM_FEATURE_LDREX
// CHECK-V4-NOT: __ARM_FEATURE_UNALIGNED
// CHECK-V4-NOT: __ARM_FEATURE_DSP
// CHECK-V4-NOT: __ARM_FEATURE_SAT
// CHECK-V4-NOT: __ARM_FEATURE_QBIT
// CHECK-V4-NOT: __ARM_FEATURE_SIMD32
// CHECK-V4-NOT: __ARM_FEATURE_IDIV
// CHECK-V4:     __ARM_ARCH 4
// CHECK-V4:     __ARM_ARCH_ISA_ARM 1

// RUN: %clang -target arm-none-linux-eabi -march=armv4t -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V4T

// CHECK-V4T: __ARM_ARCH_ISA_THUMB 1

// RUN: %clang -target arm-none-linux-eabi -march=armv5t -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V5

// CHECK-V5-NOT: __ARM_ARCH_PROFILE
// CHECK-V5-NOT: __ARM_FEATURE_LDREX
// CHECK-V5-NOT: __ARM_FEATURE_UNALIGNED
// CHECK-V5-NOT: __ARM_FEATURE_DSP
// CHECK-V5-NOT: __ARM_FEATURE_SAT
// CHECK-V5-NOT: __ARM_FEATURE_QBIT
// CHECK-V5-NOT: __ARM_FEATURE_SIMD32
// CHECK-V5-NOT: __ARM_FEATURE_IDIV
// CHECK-V5:     __ARM_ARCH 5
// CHECK-V5:     __ARM_ARCH_ISA_ARM 1
// CHECK-V5:     __ARM_ARCH_ISA_THUMB 1
// CHECK-V5:     __ARM_FEATURE_CLZ 1

// RUN: %clang -target arm-none-linux-eabi -march=armv5te -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V5E

// CHECK-V5E: __ARM_FEATURE_DSP 1
// CHECK-V5E: __ARM_FEATURE_QBIT 1

// RUN: %clang -target armv6-none-netbsd-eabi -mcpu=arm1136jf-s -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V6

// CHECK-V6-NOT: __ARM_ARCH_PROFILE
// CHECK-V6-NOT: __ARM_FEATURE_IDIV
// CHECK-V6:     __ARM_ARCH 6
// CHECK-V6:     __ARM_ARCH_ISA_ARM 1
// CHECK-V6:     __ARM_ARCH_ISA_THUMB 1
// CHECK-V6:     __ARM_FEATURE_CLZ 1
// CHECK-V6:     __ARM_FEATURE_DSP 1
// CHECK-V6:     __ARM_FEATURE_LDREX 0x4
// CHECK-V6:     __ARM_FEATURE_QBIT 1
// CHECK-V6:     __ARM_FEATURE_SAT 1
// CHECK-V6:     __ARM_FEATURE_SIMD32 1
// CHECK-V6:     __ARM_FEATURE_UNALIGNED 1

// RUN: %clang -target arm-none-linux-eabi -march=armv6m -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V6M

// CHECK-V6M-NOT: __ARM_ARCH_ISA_ARM
// CHECK-V6M-NOT: __ARM_FEATURE_CLZ
// CHECK-V6M-NOT: __ARM_FEATURE_LDREX
// CHECK-V6M-NOT: __ARM_FEATURE_UNALIGNED
// CHECK-V6M-NOT: __ARM_FEATURE_DSP
// CHECK-V6M-NOT: __ARM_FEATURE_QBIT
// CHECK-V6M-NOT: __ARM_FEATURE_SAT
// CHECK-V6M-NOT: __ARM_FEATURE_SIMD32
// CHECK-V6M-NOT: __ARM_FEATURE_IDIV
// CHECK-V6M:     __ARM_ARCH 6
// CHECK-V6M:     __ARM_ARCH_ISA_THUMB 1
// CHECK-V6M:     __ARM_ARCH_PROFILE 'M'

// RUN: %clang -target arm-none-linux-eabi -march=armv6t2 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V6T2

// CHECK-V6T2: __ARM_ARCH_ISA_THUMB 2

// RUN: %clang -target arm-none-linux-eabi -march=armv6k -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V6K
 
// CHECK-V6K: __ARM_FEATURE_LDREX 0xF

// RUN: %clang -target arm-none-linux-eabi -march=armv7-a -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A

// CHECK-V7A: __ARM_ARCH 7
// CHECK-V7A: __ARM_ARCH_ISA_ARM 1
// CHECK-V7A: __ARM_ARCH_ISA_THUMB 2
// CHECK-V7A: __ARM_ARCH_PROFILE 'A'
// CHECK-V7A: __ARM_FEATURE_CLZ 1
// CHECK-V7A: __ARM_FEATURE_DSP 1
// CHECK-V7A: __ARM_FEATURE_LDREX 0xF
// CHECK-V7A: __ARM_FEATURE_QBIT 1
// CHECK-V7A: __ARM_FEATURE_SAT 1
// CHECK-V7A: __ARM_FEATURE_SIMD32 1
// CHECK-V7A: __ARM_FEATURE_UNALIGNED 1

// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-IDIV

// CHECK-V7A-IDIV: __ARM_FEATURE_IDIV 1

// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-NO-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-NO-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7A-NO-IDIV

// CHECK-V7A-NO-IDIV-NOT: __ARM_FEATURE_IDIV

// RUN: %clang -target arm-none-linux-eabi -march=armv7-r -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7R

// CHECK-V7R: __ARM_ARCH 7
// CHECK-V7R: __ARM_ARCH_ISA_ARM 1
// CHECK-V7R: __ARM_ARCH_ISA_THUMB 2
// CHECK-V7R: __ARM_ARCH_PROFILE 'R'
// CHECK-V7R: __ARM_FEATURE_CLZ 1
// CHECK-V7R: __ARM_FEATURE_DSP 1
// CHECK-V7R: __ARM_FEATURE_LDREX 0xF
// CHECK-V7R: __ARM_FEATURE_QBIT 1
// CHECK-V7R: __ARM_FEATURE_SAT 1
// CHECK-V7R: __ARM_FEATURE_SIMD32 1
// CHECK-V7R: __ARM_FEATURE_UNALIGNED 1

// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-r4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7R-NO-IDIV

// CHECK-V7R-NO-IDIV-NOT: __ARM_FEATURE_IDIV

// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7R-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-r7 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7R-IDIV
// RUN: %clang -target arm-none-linux-eabi -mcpu=cortex-r8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7R-IDIV

// CHECK-V7R-IDIV: __ARM_FEATURE_IDIV 1

// RUN: %clang -target arm-none-linux-eabi -march=armv7-m -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7M

// CHECK-V7M-NOT: __ARM_ARCH_ISA_ARM
// CHECK-V7M-NOT: __ARM_FEATURE_DSP
// CHECK-V7M-NOT: __ARM_FEATURE_SIMD32
// CHECK-V7M:     __ARM_ARCH 7
// CHECK-V7M:     __ARM_ARCH_ISA_THUMB 2
// CHECK-V7M:     __ARM_ARCH_PROFILE 'M'
// CHECK-V7M:     __ARM_FEATURE_CLZ 1
// CHECK-V7M:     __ARM_FEATURE_IDIV 1
// CHECK-V7M:     __ARM_FEATURE_LDREX 0x7
// CHECK-V7M:     __ARM_FEATURE_QBIT 1
// CHECK-V7M:     __ARM_FEATURE_SAT 1
// CHECK-V7M:     __ARM_FEATURE_UNALIGNED 1

// RUN: %clang -target arm-none-linux-eabi -march=armv7e-m -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V7EM

// CHECK-V7EM: __ARM_FEATURE_DSP 1
// CHECK-V7EM: __ARM_FEATURE_SIMD32 1

// RUN: %clang -target arm-none-linux-eabi -march=armv8-a -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-V8A

// CHECK-V8A: __ARM_ARCH 8
// CHECK-V8A: __ARM_ARCH_ISA_ARM 1
// CHECK-V8A: __ARM_ARCH_ISA_THUMB 2
// CHECK-V8A: __ARM_ARCH_PROFILE 'A'
// CHECK-V8A: __ARM_FEATURE_CLZ 1
// CHECK-V8A: __ARM_FEATURE_DSP 1
// CHECK-V8A: __ARM_FEATURE_IDIV 1
// CHECK-V8A: __ARM_FEATURE_LDREX 0xF
// CHECK-V8A: __ARM_FEATURE_QBIT 1
// CHECK-V8A: __ARM_FEATURE_SAT 1
// CHECK-V8A: __ARM_FEATURE_SIMD32 1
// CHECK-V8A: __ARM_FEATURE_UNALIGNED 1

