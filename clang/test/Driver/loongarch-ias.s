/// This file checks options are correctly passed to cc1as for LoongArch targets.

/// Check `-target-abi`.
// RUN: %clang --target=loongarch32 -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=ilp32d --check-prefix=ABI %s
// RUN: %clang --target=loongarch32 -mabi=ilp32d -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=ilp32d --check-prefix=ABI %s
// RUN: %clang --target=loongarch32 -mabi=ilp32f -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=ilp32f --check-prefix=ABI %s
// RUN: %clang --target=loongarch32 -mabi=ilp32s -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=ilp32s --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64d --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64d -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64d --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64f -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64f --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64s -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64s --check-prefix=ABI %s

// ALL: -cc1as

// ABI: "-target-abi" "[[ABI]]"
