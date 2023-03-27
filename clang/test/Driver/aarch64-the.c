// Test that target feature the is implemented and available correctly

// FEAT_THE is optional (off by default) for v8.9a/9.4a, and can be disabled using +nothe
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.9-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.9-a+the   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.9-a+nothe %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.4-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.4-a+the   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.4-a+nothe %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// FEAT_THE is optional (off by default) for v8.8a/9.3a and older, and can be enabled using +the
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a+the   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a+nothe %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a+the   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a+nothe %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// FEAT_THE is invalid before v8
// RUN: %clang -### --target=arm-none-none-eabi -march=armv7-a+the     %s 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error: unsupported argument 'armv7-a+the' to option '-march='
// ENABLED: "-target-feature" "+the"
// NOT_ENABLED-NOT: "-target-feature" "+the"
// DISABLED: "-target-feature" "-the"

