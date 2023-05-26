// Test that target feature ite is implemented and available correctly

// FEAT_ITE is optional (off by default) for v8.9a/9.4a and older, and can be enabled using +ite
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a+ite   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.8-a+noite %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a       %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a+ite   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv9.3-a+noite %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// FEAT_ITE is invalid before v8
// RUN: %clang -### --target=arm-none-none-eabi -march=armv7-a+ite     %s 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error: unsupported argument 'armv7-a+ite' to option '-march='
// ENABLED: "-target-feature" "+ite"
// NOT_ENABLED-NOT: "-target-feature" "+ite"
// DISABLED: "-target-feature" "-ite"
