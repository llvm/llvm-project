// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+dsp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-DSP < %t %s
// CHECK-DSP: "-target-feature" "+dsp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVE < %t %s
// CHECK-MVE: "-target-feature" "+mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP < %t %s
// CHECK-MVEFP: "-target-feature" "+mve.fp"
// CHECK-MVEFP-NOT: "-target-feature" "+fp64"

double foo (double a) { return a; }
