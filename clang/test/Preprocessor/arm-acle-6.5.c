// RUN: %clang -target arm-eabi -mfpu=none -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP
// RUN: %clang -target armv4-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP
// RUN: %clang -target armv5-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP
// RUN: %clang -target armv6m-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP
// RUN: %clang -target armv7r-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP
// RUN: %clang -target armv7m-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FP

// CHECK-NO-FP-NOT: __ARM_FP 0x{{.*}}

// RUN: %clang -target arm-eabi -mfpu=vfpv3xd -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-ONLY

// CHECK-SP-ONLY: __ARM_FP 0x4

// RUN: %clang -target arm-eabi -mfpu=vfpv3xd-fp16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-HP
// RUN: %clang -target arm-eabi -mfpu=fpv4-sp-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-HP
// RUN: %clang -target arm-eabi -mfpu=fpv5-sp-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-HP

// CHECK-SP-HP: __ARM_FP 0x6

// RUN: %clang -target arm-eabi -mfpu=vfp -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfpv2 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfpv3 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfp3-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target armv6-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target armv7a-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target armv7ve-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP

// CHECK-SP-DP: __ARM_FP 0xC

// RUN: %clang -target arm-eabi -mfpu=vfpv3-fp16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=vfpv3-d16-fp16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=vfpv4-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=fpv5-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=neon-fp16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=neon-vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP

// CHECK-SP-DP-HP: __ARM_FP 0xE

// RUN: %clang -target armv4-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv5-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv6-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv6m-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv7m-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA

// CHECK-NO-FMA-NOT: __ARM_FEATURE_FMA

// RUN: %clang -target armv7a-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv7a-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-FMA
// RUN: %clang -target armv7ve-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv7ve-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-FMA
// RUN: %clang -target armv7r-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv7r-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-FMA
// RUN: %clang -target armv7em-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-FMA
// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-FMA
// RUN: %clang -target armv8-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-FMA

// CHECK-FMA: __ARM_FEATURE_FMA 1

// RUN: %clang -target armv4-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-NEON
// RUN: %clang -target armv5-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-NEON
// RUN: %clang -target armv6-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-NEON

// CHECK-NO-NEON-NOT: __ARM_NEON
// CHECK-NO-NEON-NOT: __ARM_NEON_FP 0x{{.*}}

// RUN: %clang -target armv7-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NEON-SP

// CHECK-NEON-SP: __ARM_NEON 1
// CHECK-NEON-SP: __ARM_NEON_FP 0x4

// RUN: %clang -target armv7-eabi -mfpu=neon-fp16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NEON-SP-HP
// RUN: %clang -target armv7-eabi -mfpu=neon-vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NEON-SP-HP
// RUN: %clang -target armv7-eabi -mfpu=neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NEON-SP-HP
// RUN: %clang -target armv7-eabi -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NEON-SP-HP

// CHECK-NEON-SP-HP: __ARM_NEON 1
// CHECK-NEON-SP-HP: __ARM_NEON_FP 0x6

// RUN: %clang -target armv4-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-EXTENSIONS
// RUN: %clang -target armv5-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-EXTENSIONS
// RUN: %clang -target armv6-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-EXTENSIONS
// RUN: %clang -target armv7-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-NO-EXTENSIONS

// CHECK-NO-EXTENSIONS-NOT: __ARM_FEATURE_CRC32
// CHECK-NO-EXTENSIONS-NOT: __ARM_FEATURE_CRYPTO
// CHECK-NO-EXTENSIONS-NOT: __ARM_FEATURE_DIRECTED_ROUNDING
// CHECK-NO-EXTENSIONS-NOT: __ARM_FEATURE_NUMERIC_MAXMIN

// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-EXTENSIONS

// CHECK-EXTENSIONS: __ARM_FEATURE_CRC32 1
// CHECK-EXTENSIONS: __ARM_FEATURE_CRYPTO 1
// CHECK-EXTENSIONS: __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-EXTENSIONS: __ARM_FEATURE_NUMERIC_MAXMIN 1

