// RUN: not clang-check "%s" -- -no-integrated-as -c 2>&1 | FileCheck %s
// The following test uses multiple time the same '-no-integrated-as' flag in order to make sure those flags are really skipped, and not just overwritten by luck :
// RUN: not clang-check "%s" -- -target x86_64-win32 -c -no-integrated-as -no-integrated-as -no-integrated-as 2>&1 | FileCheck %s

// CHECK: a type specifier is required
invalid;
