// Test that target feature d128 is implemented and available correctly

// FEAT_D128 is optional (off by default) for v9.4a and older, and can be enabled using +d128
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a        %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a+d128   %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### --target=aarch64-none-elf -march=armv9.4-a+nod128 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED

// ENABLED: "-target-feature" "+d128"
// NOT_ENABLED-NOT: "-target-feature" "+d128"
