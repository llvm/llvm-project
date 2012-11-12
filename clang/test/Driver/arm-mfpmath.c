// Test different values of -mfpmath.

// RUN: %clang -target arm-apple-darwin10 -mfpmath=vfp %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP %s
// CHECK-VFP: "-target-feature" "-neonfp"

// RUN: %clang -target arm-apple-darwin10 -mfpmath=vfp2 %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP2 %s
// CHECK-VFP2: "-target-feature" "-neonfp"

// RUN: %clang -target arm-apple-darwin10 -mfpmath=vfp3 %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// CHECK-VFP3: "-target-feature" "-neonfp"

// RUN: %clang -target arm-apple-darwin10 -mfpmath=vfp4 %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// CHECK-VFP4: "-target-feature" "-neonfp"

// RUN: %clang -target arm-apple-darwin10 -mfpmath=neon %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neonfp"

// RUN: %clang -target arm-apple-darwin10 -mfpmath=foo %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ERROR %s
// CHECK-ERROR: clang compiler does not support '-mfpmath=foo'

// RUN: %clang -target arm-apple-darwin10 -mcpu=arm1136j-s -mfpmath=neon %s -### -c -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-ERROR %s
// CHECK-MCPU-ERROR: error: invalid feature '-mfpmath=neon' for CPU 'arm1136j-s'
