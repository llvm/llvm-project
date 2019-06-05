// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+dsp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-DSP < %t %s
// CHECK-DSP: "-target-feature" "+dsp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FP < %t %s
// CHECK-FP: "-target-feature" "+fp-armv8"
// CHECK-FP-NOT: "-target-feature" "+fp64"
// CHECK-FP-NOT: "-target-feature" "+d32"
// CHECK-FP: "-target-feature" "+fullfp16"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nofp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOFP < %t %s
// CHECK-NOFP: "-target-feature" "-vfp2" "-target-feature" "-vfp3" "-target-feature" "-fp16" "-target-feature" "-vfp4" "-target-feature" "-fp-armv8" "-target-feature" "-fp64" "-target-feature" "-d32" "-target-feature" "-neon" "-target-feature" "-crypto"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FPDP < %t %s
// CHECK-FPDP: "-target-feature" "+fp-armv8"
// CHECK-FPDP: "-target-feature" "+fullfp16"
// CHECK-FPDP: "-target-feature" "+fp64"
// CHECK-FPDP-NOT: "-target-feature" "+d32"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nofp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOFPDP < %t %s
// CHECK-NOFPDP: "-target-feature" "-fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVE < %t %s
// CHECK-MVE: "-target-feature" "+mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nomve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOMVE < %t %s
// CHECK-NOMVE: "-target-feature" "-mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP < %t %s
// CHECK-MVEFP: "-target-feature" "+mve.fp"
// CHECK-MVEFP-NOT: "-target-feature" "+fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nomve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOMVEFP < %t %s
// CHECK-NOMVEFP: "-target-feature" "-mve.fp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP_DP < %t %s
// CHECK-MVEFP_DP: "-target-feature" "+mve.fp"
// CHECK-MVEFP_DP: "-target-feature" "+fp64"

double foo (double a) { return a; }
