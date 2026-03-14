/// Check handling the CPU is 64bit while ABI is O32.
/// when build for MIPS platforms.

/// abi-n32
// RUN: %clang -### -c %s --target=mips-linux-gnu -mabi=n32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-N32 %s
// CHECK-ABI-N32: "-target-abi" "n32"

/// abi-64
// RUN: %clang -### -c %s --target=mips-linux-gnu -mabi=64 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-64 %s
// CHECK-ABI-64: "-target-abi" "n64"


/// -march=mips3
// RUN: %clang -### -c %s --target=mips-linux-gnu -march=mips3 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPS3 %s
// CHECK-MIPS-MIPS3: "-target-cpu" "mips3" {{.*}} "-target-abi" "o32"

/// -march=mips4
// RUN: %clang -### -c %s --target=mips-linux-gnu -march=mips4 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPS4 %s
// CHECK-MIPS-MIPS4: "-target-cpu" "mips4" {{.*}} "-target-abi" "o32"

/// FIXME: MIPS V is not implemented yet.

/// -march=mips64
/// RUN: %clang -### -c %s --target=mips-linux-gnu -march=mips64 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPS64 %s
// CHECK-MIPS-MIPS64: "-target-cpu" "mips64" {{.*}} "-target-abi" "o32"

/// -march=mips64r2
/// RUN: %clang -### -c %s --target=mips-linux-gnu -march=mips64r2 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPS64R2 %s
// CHECK-MIPS-MIPS64R2: "-target-cpu" "mips64r2" {{.*}} "-target-abi" "o32"

/// -march=mips64r6
// RUN: %clang -### -c %s --target=mips-linux-gnu -march=mips64r6 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPS64R6 %s
// CHECK-MIPS-MIPS64R6: "-target-cpu" "mips64r6" {{.*}} "-target-abi" "o32"


/// mipsisa3
// RUN: %clang -### -c %s --target=mips64-linux-gnu -march=mips3 -mabi=32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPSISA3 %s
// CHECK-MIPS-MIPSISA3: "-target-cpu" "mips3" {{.*}} "-target-abi" "o32"

/// mipsisa4
// RUN: %clang -### -c %s --target=mips64-linux-gnu -march=mips4 -mabi=32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPSISA4 %s
// CHECK-MIPS-MIPSISA4: "-target-cpu" "mips4" {{.*}} "-target-abi" "o32"

/// FIXME: MIPS V is not implemented yet.

/// mipsisa64
// RUN: %clang -### -c %s --target=mips64-linux-gnu -march=mips64 -mabi=32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPSISA64 %s
// CHECK-MIPS-MIPSISA64: "-target-cpu" "mips64" {{.*}} "-target-abi" "o32"

/// mipsisa64r2
// RUN: %clang -### -c %s --target=mips64-linux-gnu -march=mips64r2 -mabi=32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPSISA64R2 %s
// CHECK-MIPS-MIPSISA64R2: "-target-cpu" "mips64r2" {{.*}} "-target-abi" "o32"

/// mipsisa64r6
// RUN: %clang -### -c %s --target=mips64-linux-gnu -march=mips64r6 -mabi=32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-MIPSISA64R6 %s
// CHECK-MIPS-MIPSISA64R6: "-target-cpu" "mips64r6" {{.*}} "-target-abi" "o32"
