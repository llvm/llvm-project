// Make sure AddressSanitizer disables _FORTIFY_SOURCE on Darwin.

// RUN: %clang -faddress-sanitizer %s -E -dM -target x86_64-darwin - | FileCheck %s
// RUN: %clang -fsanitize=address  %s -E -dM -target x86_64-darwin - | FileCheck %s

// CHECK: #define _FORTIFY_SOURCE 0
