// XFAIL: *
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a %s 2>&1 | FileCheck -check-prefix=CHECK-NEON %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve %s 2>&1 | FileCheck -check-prefix=CHECK-SVE %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2-sha3 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2SHA3 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve+nosve %s 2>&1 | FileCheck -check-prefix=CHECK-SVE-NOSVE %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2+nosve2-sha3 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2-NOSVE2SHA3 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2-sha3+nosve2 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2SHA3-NOSVE2 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2-sha3+nosve %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2SHA3-NOSVE %s

// CHECK-NEON: "-target_features" "+neon,+v8a"
// CHECK-NEON-NOT: "-vscale_range_min"
// CHECK-NEON-NOT: "-vscale_range_max"
// CHECK-SVE: "-target_features" "+neon,+v8a,+sve"
// CHECK-SVE-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
// CHECK-SVE2: "-target_features" "+neon,+v8a,+sve2,+sve"
// CHECK-SVE2-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
// CHECK-SVE2SHA3: "-target_features" "+neon,+v8a,+sve2-sha3,+sve,+sve2"
// CHECK-SVE2SHA3-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
// CHECK-SVE-NOSVE: "-target_features" "+neon,+v8a,-sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
// CHECK-SVE-NOSVE-NOT: "-vscale_range_min"
// CHECK-SVE-NOSVE-NOT: "-vscale_range_max"
// CHECK-SVE2-NOSVE2SHA3: "-target_features" "+neon,+v8a,+sve2,+sve,-sve2-sha3"
// CHECK-SVE2-NOSVE2SHA3-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
// CHECK-SVE2SHA3-NOSVE2: "-target_features" "+neon,+v8a,+sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
// CHECK-SVE2SHA3-NOSVE2-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
// CHECK-SVE2SHA3-NOSVE: "-target_features" "+neon,+v8a,-sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
// CHECK-SVE2SHA3-NOSVE-NOT: "-vscale_range_min"
// CHECK-SVE2SHA3-NOSVE-NOT: "-vscale_range_max"
