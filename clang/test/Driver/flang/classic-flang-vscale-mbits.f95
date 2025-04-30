// XFAIL: *
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve -msve-vector-bits=128 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE-128 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve -msve-vector-bits=128+ %s 2>&1 | FileCheck -check-prefix=CHECK-SVE-128PLUS %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve -msve-vector-bits=256 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE-256 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve -msve-vector-bits=256+ %s 2>&1 | FileCheck -check-prefix=CHECK-SVE-256PLUS %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2 -msve-vector-bits=512 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2-512 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2 -msve-vector-bits=512+ %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2-512PLUS %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2-sha3 -msve-vector-bits=2048 %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2SHA3-2048 %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2-sha3 -msve-vector-bits=2048+ %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2SHA3-2048PLUS %s
// RUN: %clang --driver-mode=flang -### -S --target=aarch64 -march=armv8-a+sve2 -msve-vector-bits=scalable %s 2>&1 | FileCheck -check-prefix=CHECK-SVE2-SCALABLE %s

// CHECK-SVE-128: "-target_features" "+neon,+v8a,+sve"
// CHECK-SVE-128-DAG: "-vscale_range_min" "1" "-vscale_range_max" "1"
// CHECK-SVE-128PLUS: "-target_features" "+neon,+v8a,+sve"
// CHECK-SVE-128PLUS-DAG: "-vscale_range_min" "1" "-vscale_range_max" "0"
// CHECK-SVE-256: "-target_features" "+neon,+v8a,+sve"
// CHECK-SVE-256-DAG: "-vscale_range_min" "2" "-vscale_range_max" "2"
// CHECK-SVE-256PLUS: "-target_features" "+neon,+v8a,+sve"
// CHECK-SVE-256PLUS-DAG: "-vscale_range_min" "2" "-vscale_range_max" "0"
// CHECK-SVE2-512: "-target_features" "+neon,+v8a,+sve2,+sve"
// CHECK-SVE2-512-DAG: "-vscale_range_min" "4" "-vscale_range_max" "4"
// CHECK-SVE2-512PLUS: "-target_features" "+neon,+v8a,+sve2,+sve"
// CHECK-SVE2-512PLUS-DAG: "-vscale_range_min" "4" "-vscale_range_max" "0"
// CHECK-SVE2SHA3-2048: "-target_features" "+neon,+v8a,+sve2-sha3,+sve,+sve2"
// CHECK-SVE2SHA3-2048-DAG: "-vscale_range_min" "16" "-vscale_range_max" "16"
// CHECK-SVE2SHA3-2048PLUS: "-target_features" "+neon,+v8a,+sve2-sha3,+sve,+sve2"
// CHECK-SVE2SHA3-2048PLUS-DAG: "-vscale_range_min" "16" "-vscale_range_max" "0"
// CHECK-SVE2-SCALABLE: "-target_features" "+neon,+v8a,+sve2,+sve"
// CHECK-SVE2-SCALABLE-DAG: "-vscale_range_min" "1" "-vscale_range_max" "16"
