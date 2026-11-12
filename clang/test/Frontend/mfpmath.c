// RUN: %clang_cc1 -triple i686-pc-linux -target-feature -sse  %s

// RUN: %clang_cc1 -triple i686-pc-linux -target-feature -sse -mfpmath 387 %s

// RUN: %clang_cc1 -triple i686-pc-linux -target-feature +sse %s

// RUN: %clang_cc1 -triple i686-pc-linux -target-feature +sse -mfpmath sse %s

// RUN: not %clang_cc1 -triple i686-pc-linux -target-feature +sse \
// RUN: -mfpmath xyz %s 2>&1 | FileCheck --check-prefix=CHECK-XYZ %s
// CHECK-XYZ: error: unknown FP unit 'xyz'

// RUN: not %clang_cc1 -triple i686-pc-linux -target-feature +sse \
// RUN: -mfpmath 387 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-387 %s
// CHECK-NO-387: error: the '387' unit is not supported with this instruction set

// RUN: not %clang_cc1 -triple i686-pc-linux -target-feature -sse \
// RUN: -mfpmath sse %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SSE %s
// CHECK-NO-SSE: error: the 'sse' unit is not supported with this instruction set


// RUN: %clang_cc1 -triple arm-apple-darwin10 -mfpmath vfp %s

// RUN: %clang_cc1 -triple arm-apple-darwin10 -mfpmath vfp2 %s

// RUN: %clang_cc1 -triple arm-apple-darwin10 -mfpmath vfp3 %s

// RUN: %clang_cc1 -triple arm-apple-darwin10 -mfpmath vfp4 %s

// RUN: %clang_cc1 -triple arm-apple-darwin10 -target-cpu cortex-a9 \
// RUN: -mfpmath neon %s

// RUN: not %clang_cc1 -triple arm-apple-darwin10 -mfpmath foo %s 2>&1 \
// RUN: FileCheck --check-prefix=CHECK-FOO %s
// CHECK-FOO: unknown FP unit 'foo'

// RUN: not %clang_cc1 -triple arm-apple-darwin10 -target-cpu arm1136j-s \
// RUN: -mfpmath neon %s 2>&1 | FileCheck --check-prefix=CHECK-NO-NEON %s

// RUN: not %clang_cc1 -triple arm-apple-darwin10 -target-cpu cortex-a9 \
// RUN: -target-feature -neon -mfpmath neon %s 2>&1 | FileCheck --check-prefix=CHECK-NO-NEON %s

// CHECK-NO-NEON: error: the 'neon' unit is not supported with this instruction set
