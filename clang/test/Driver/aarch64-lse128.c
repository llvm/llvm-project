// Test that target feature lse128 is implemented and available correctly

// FEAT_LSE128 is optional (off by default) for v9.4a and older, and can be enabled using +lse128
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.4-a          %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.4-a+lse128   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.4-a+nolse128 %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// ENABLED: "-target-feature" "+lse128"
// NOT_ENABLED-NOT: "-target-feature" "+lse128"
// DISABLED: "-target-feature" "-lse128"
