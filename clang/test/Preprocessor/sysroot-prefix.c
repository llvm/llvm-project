// RUN: %clang_cc1 -v -isysroot /var/empty -I /var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_NO_SYSROOT %s
// RUN: %clang_cc1 -v -isysroot /var/empty -I =/var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_SYSROOT_DEV_NULL %s
// RUN: %clang_cc1 -v -I =/var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-NO_ISYSROOT_SYSROOT_DEV_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_SYSROOT_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty -isysroot /var/empty/root -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSROOT_SYSROOT_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty/root -isysroot /var/empty -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSROOT_SWAPPED_SYSROOT_NULL %s

// CHECK-ISYSROOT_NO_SYSROOT: ignoring nonexistent directory "/var/empty/include"
// CHECK-ISYSROOT_NO_SYSROOT-NOT: ignoring nonexistent directory "/var/empty/var/empty/include"

// CHECK-ISYSROOT_SYSROOT_DEV_NULL: ignoring nonexistent directory "/var/empty/var/empty/include"
// CHECK-ISYSROOT_SYSROOT_DEV_NULL-NOT: ignoring nonexistent directory "/var/empty"

// CHECK-NO_ISYSROOT_SYSROOT_DEV_NULL: ignoring nonexistent directory "=/var/empty/include"
// CHECK-NO_ISYSROOT_SYSROOT_DEV_NULL-NOT: ignoring nonexistent directory "/var/empty/include"

// CHECK-ISYSROOT_SYSROOT_NULL: ignoring nonexistent directory "/var/empty{{.}}null"
// CHECK-ISYSROOT_SYSROOT_NULL-NOT: ignoring nonexistent directory "=null"

// CHECK-ISYSROOT_ISYSROOT_SYSROOT_NULL: ignoring nonexistent directory "/var/empty/root{{.}}null"
// CHECK-ISYSROOT_ISYSROOT_SYSROOT_NULL-NOT: ignoring nonexistent directory "=null"

// CHECK-ISYSROOT_ISYSROOT_SWAPPED_SYSROOT_NULL: ignoring nonexistent directory "/var/empty{{.}}null"
// CHECK-ISYSROOT_ISYSROOT_SWAPPED_SYSROOT_NULL-NOT: ignoring nonexistent directory "=null"

