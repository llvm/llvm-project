// Check that -mfloat-abi=hard gives a warning if FP registers aren't available.
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m0 -mfloat-abi=hard -### -c %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m0 -mhard-float -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HARDFLOAT %s

// -mfloat-abi=hard and -march=...+nofp are incompatible in this instance:
// RUN: %clang --target=arm-none-eabi -march=armv8.1-m.main+nofp -mfloat-abi=hard -### -c %s 2>&1
// -mfloat-abi=hard and -march=...+nofp are compatible in this instance:
// RUN: %clang --target=arm-none-eabi -march=armv8.1-m.main+mve+nofp -mfloat-abi=hard -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NOWARN %s

// Here the float ABI is calculated as "hard" and FP registers are
// calculated to not be available. Since the float ABI wasn't specified
// explicitly, the warning should not be emitted.
// RUN: not %clang --target=thumbv5-windows -mcpu=arm10tdmi -### -c %s -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=NOWARN %s

// CHECK: warning: '-mfloat-abi=hard': selected processor lacks floating point registers
// HARDFLOAT: warning: '-mhard-float': selected processor lacks floating point registers
// NOWARN-NOT: selected processor lacks floating point registers
