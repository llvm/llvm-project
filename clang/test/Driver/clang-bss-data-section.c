// RUN: %clang -### -c --target=arm-none-eabi \
// RUN:   -fclang-bss-section=.my_bss -fclang-data-section=.my_data %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-fclang-bss-section=.my_bss"
// CHECK-SAME: "-fclang-data-section=.my_data"

// The last occurrence wins.
// RUN: %clang -### -c --target=arm-none-eabi \
// RUN:   -fclang-bss-section=.first -fclang-bss-section=.second %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LAST

// LAST-NOT: "-fclang-bss-section=.first"
// LAST: "-fclang-bss-section=.second"

// RUN: %clang -### -c --target=arm-none-eabi %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE

// NONE-NOT: "-fclang-bss-section=
// NONE-NOT: "-fclang-data-section=