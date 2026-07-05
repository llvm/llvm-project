// Test that target feature FEAT_RCPC3 is implemented and available correctly

// FEAT_RCPC3 is optional for v8.2a onwards, and can be enabled with +rcpc3
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.9-a         %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.9-a+rcpc3   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.9-a+norcpc3 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a         %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a+rcpc3   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a+norcpc3 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED

// FEAT_RCPC3 is optional (off by default) for v8.8a/9.3a and older, and can be enabled using +rcpc3
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2-a         %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2-a+rcpc3   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2-a+norcpc3 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9-a         %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9-a+rcpc3   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9-a+norcpc3 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED

// FEAT_RCPC3 is invalid before v8
// RUN: not %clang -### --target=arm-none-none-eabi -march=armv7-a+rcpc3         %s 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error: unsupported argument 'armv7-a+rcpc3' to option '-march='
// ENABLED: "-target-feature" "+rcpc3"
// NOT_ENABLED-NOT: "-target-feature" "+rcpc3"

