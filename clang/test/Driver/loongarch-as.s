/// This file checks options are correctly passed to as for LoongArch targets.

/// Check `-mabi`.
// RUN: %clang --target=loongarch64 -### -fno-integrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64d --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64d -### -fno-integrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64d --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64f -### -fno-integrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64f --check-prefix=ABI %s
// RUN: %clang --target=loongarch64 -mabi=lp64s -### -fno-integrated-as -c %s 2>&1 | \
// RUN:   FileCheck -DABI=lp64s --check-prefix=ABI %s

// ALL: as

// ABI: "-mabi=[[ABI]]"
