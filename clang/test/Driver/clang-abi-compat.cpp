// PS4 target requires clang ABI version 6, check that a warning is emitted when a version other than 6 is requested.
// RUN: %clang -S --target=x86_64-scei-ps4 -fclang-abi-compat=4 %s 2>&1 | FileCheck %s -check-prefix=CHECK-WARNING
// RUN: %clang -S --target=x86_64-scei-ps4 -fclang-abi-compat=latest %s 2>&1 | FileCheck %s -check-prefix=CHECK-WARNING

// REQUIRES: x86-registered-target

// CHECK-WARNING: warning: target requires clang ABI version 6, ignoring requested version

