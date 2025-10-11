// Test the -fsanitize-address-disable-container-overflow option

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s \
// RUN:   -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT-NOT: -fsanitize-address-disable-container-overflow

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address \
// RUN:   -fsanitize-address-disable-container-overflow %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-ENABLE %s
// CHECK-ENABLE: "-fsanitize-address-disable-container-overflow"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address \
// RUN:   -fno-sanitize-address-disable-container-overflow %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-DISABLE %s
// CHECK-DISABLE-NOT: -fsanitize-address-disable-container-overflow

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address \
// RUN:   -fsanitize-address-disable-container-overflow \
// RUN:   -fno-sanitize-address-disable-container-overflow %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-OVERRIDE %s
// CHECK-OVERRIDE-NOT: -fsanitize-address-disable-container-overflow

// Test that the flag generates unused warning without address sanitizer
// RUN: %clang -target x86_64-linux-gnu -fsanitize-address-disable-container-overflow %s \
// RUN:   -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-NO-ASAN %s
// CHECK-NO-ASAN: warning: argument unused during compilation: '-fsanitize-address-disable-container-overflow'

// Test with kernel address sanitizer
// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address \
// RUN:   -fsanitize-address-disable-container-overflow %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-KERNEL-ASAN %s  
// CHECK-KERNEL-ASAN: "-fsanitize-address-disable-container-overflow"