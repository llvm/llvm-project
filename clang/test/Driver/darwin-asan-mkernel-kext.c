// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=address -mkernel -### \
// RUN:   %s 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=address -fapple-kext \
// RUN:   -### %s 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=address -fapple-kext \
// RUN:   -mkernel -### %s 2>&1 | FileCheck %s

// CHECK: "-fsanitize-address-destructor-kind=none"

// Check it's possible to override the driver's decision.
// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=address -fapple-kext \
// RUN:   -mkernel -### -fsanitize-address-destructor-kind=global %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-OVERRIDE %s

// CHECK-OVERRIDE: "-fsanitize-address-destructor-kind=global"
