// Test that different values of -mfpu pick correct CSKY FPU target-feature(s).

// RUN: %clang -target csky-unknown-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+hard-float"
// CHECK-DEFAULT-NOT: "-target-feature" "+hard-float-abi"
// CHECK-DEFAULT-NOT: "-target-feature" "+fpuv2_sf"
// CHECK-DEFAULT-NOT: "-target-feature" "+fpuv2_df"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv2 -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV2 %s
// CHECK-FPV2-NOT: "-target-feature" "+hard-float"
// CHECK-FPV2-NOT: "-target-feature" "+hard-float-abi"
// CHECK-FPV2: "-target-feature" "+fpuv2_sf"
// CHECK-FPV2: "-target-feature" "+fpuv2_df"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv2 -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV2-HARD %s
// CHECK-FPV2-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV2-HARD: "-target-feature" "+hard-float"
// CHECK-FPV2-HARD: "-target-feature" "+fpuv2_sf"
// CHECK-FPV2-HARD: "-target-feature" "+fpuv2_df"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv2_divd -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV2DIVD-HARD %s
// CHECK-FPV2DIVD-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV2DIVD-HARD: "-target-feature" "+hard-float"
// CHECK-FPV2DIVD-HARD: "-target-feature" "+fpuv2_sf"
// CHECK-FPV2DIVD-HARD: "-target-feature" "+fpuv2_df"
// CHECK-FPV2DIVD-HARD: "-target-feature" "+fdivdu"

// RUN: %clang -target csky-unknown-linux %s -mfpu=auto -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-AUTO-HARD %s
// CHECK-AUTO-HARD: "-target-feature" "+hard-float-abi"
// CHECK-AUTO-HARD: "-target-feature" "+hard-float"
// CHECK-AUTO-HARD: "-target-feature" "+fpuv2_sf"
// CHECK-AUTO-HARD: "-target-feature" "+fpuv2_df"
// CHECK-AUTO-HARD: "-target-feature" "+fdivdu"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv2_sf -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV2SF-HARD %s
// CHECK-FPV2SF-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV2SF-HARD: "-target-feature" "+hard-float"
// CHECK-FPV2SF-HARD: "-target-feature" "+fpuv2_sf"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv3 -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV3-HARD %s
// CHECK-FPV3-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV3-HARD: "-target-feature" "+hard-float"
// CHECK-FPV3-HARD: "-target-feature" "+fpuv3_hf"
// CHECK-FPV3-HARD: "-target-feature" "+fpuv3_hi"
// CHECK-FPV3-HARD: "-target-feature" "+fpuv3_sf"
// CHECK-FPV3-HARD: "-target-feature" "+fpuv3_df"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv3_hf -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV3HF-HARD %s
// CHECK-FPV3HF-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV3HF-HARD: "-target-feature" "+hard-float"
// CHECK-FPV3HF-HARD: "-target-feature" "+fpuv3_hf"
// CHECK-FPV3HF-HARD: "-target-feature" "+fpuv3_hi"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv3_hsf -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV3HSF-HARD %s
// CHECK-FPV3HSF-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV3HSF-HARD: "-target-feature" "+hard-float"
// CHECK-FPV3HSF-HARD: "-target-feature" "+fpuv3_hf"
// CHECK-FPV3HSF-HARD: "-target-feature" "+fpuv3_hi"
// CHECK-FPV3HSF-HARD: "-target-feature" "+fpuv3_sf"

// RUN: %clang -target csky-unknown-linux %s -mfpu=fpv3_sdf -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV3DF-HARD %s
// CHECK-FPV3DF-HARD: "-target-feature" "+hard-float-abi"
// CHECK-FPV3DF-HARD: "-target-feature" "+hard-float"
// CHECK-FPV3DF-HARD: "-target-feature" "+fpuv3_sf"
// CHECK-FPV3DF-HARD: "-target-feature" "+fpuv3_df"

// RUN: %clang -target csky-unknown-linux %s -mcpu=c810 -mfpu=fpv3 -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV3-C810 %s
// CHECK-FPV3-C810: "-target-feature" "+hard-float-abi"
// CHECK-FPV3-C810: "-target-feature" "+hard-float"
// CHECK-FPV3-C810: "-target-feature" "+fpuv3_hf"
// CHECK-FPV3-C810: "-target-feature" "+fpuv3_hi"
// CHECK-FPV3-C810: "-target-feature" "+fpuv3_sf"
// CHECK-FPV3-C810: "-target-feature" "+fpuv3_df"
// CHECK-FPV3-C810-NOT: "-target-feature" "+fpuv2"

// RUN: %clang -target csky-unknown-linux %s -mcpu=c860 -mfpu=fpv2 -mhard-float -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPV2-C860 %s
// CHECK-FPV2-C860: "-target-feature" "+hard-float-abi"
// CHECK-FPV2-C860: "-target-feature" "+hard-float"
// CHECK-FPV2-C860: "-target-feature" "+fpuv2_sf"
// CHECK-FPV2-C860: "-target-feature" "+fpuv2_df"
// CHECK-FPV2-C860-NOT: "-target-feature" "+fpuv3"