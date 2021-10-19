// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+dsp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-DSP < %t %s
// CHECK-DSP: "-target-feature" "+dsp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FP < %t %s
// CHECK-FP-DAG: "-target-feature" "+fp-armv8d16sp"
// CHECK-FP-NOT: "-target-feature" "+fp-armv8d16"
// CHECK-FP-NOT: "-target-feature" "+fp-armv8sp"
// CHECK-FP-NOT: "-target-feature" "+fp-armv8"
// CHECK-FP-NOT: "-target-feature" "+fp64"
// CHECK-FP-NOT: "-target-feature" "+d32"
// CHECK-FP-DAG: "-target-feature" "+fullfp16"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nofp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOFP < %t %s
// CHECK-NOFP-DAG: "-target-feature" "-vfp2"
// CHECK-NOFP-DAG: "-target-feature" "-vfp3"
// CHECK-NOFP-DAG: "-target-feature" "-fp16"
// CHECK-NOFP-DAG: "-target-feature" "-vfp4"
// CHECK-NOFP-DAG: "-target-feature" "-fp-armv8"
// CHECK-NOFP-DAG: "-target-feature" "-fp64"
// CHECK-NOFP-DAG: "-target-feature" "-d32"
// CHECK-NOFP-DAG: "-target-feature" "-neon"
// CHECK-NOFP-DAG: "-target-feature" "-sha2"
// CHECK-NOFP-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FPDP < %t %s
// CHECK-FPDP-NOT: "-target-feature" "+fp-armv8sp"
// CHECK-FPDP-DAG: "-target-feature" "+fp-armv8d16"
// CHECK-FPDP-NOT: "-target-feature" "+fp-armv8"
// CHECK-FPDP-DAG: "-target-feature" "+fullfp16"
// CHECK-FPDP-DAG: "-target-feature" "+fp64"
// CHECK-FPDP-NOT: "-target-feature" "+d32"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nofp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOFPDP < %t %s
// CHECK-NOFPDP-DAG: "-target-feature" "-fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVE < %t %s
// CHECK-MVE-DAG: "-target-feature" "+mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nomve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOMVE < %t %s
// CHECK-NOMVE-DAG: "-target-feature" "-mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP < %t %s
// CHECK-MVEFP-DAG: "-target-feature" "+mve.fp"
// CHECK-MVEFP-NOT: "-target-feature" "+fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nomve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOMVEFP < %t %s
// CHECK-NOMVEFP: "-target-feature" "-mve.fp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP_DP < %t %s
// CHECK-MVEFP_DP-DAG: "-target-feature" "+mve.fp"
// CHECK-MVEFP_DP-DAG: "-target-feature" "+fp64"

double foo (double a) { return a; }
