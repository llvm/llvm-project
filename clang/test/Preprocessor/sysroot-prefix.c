// RUN: %clang_cc1 -v -isysroot /var/empty -I /var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_NO_SYSROOT %s
// RUN: %clang_cc1 -v -isysroot /var/empty -I =/var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_SYSROOT_DEV_NULL %s
// RUN: %clang_cc1 -v -I =/var/empty/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-NO_ISYSROOT_SYSROOT_DEV_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_SYSROOT_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty -isysroot /var/empty/root -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSROOT_SYSROOT_NULL %s
// RUN: %clang_cc1 -v -isysroot /var/empty/root -isysroot /var/empty -I =null -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSROOT_SWAPPED_SYSROOT_NULL %s
// RUN: %clang_cc1 -v -isystem=/usr/include -isysroot /var/empty -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSTEM_SYSROOT %s
// RUN: %clang_cc1 -v -isystem=/usr/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_ISYSTEM_NO_SYSROOT %s
// RUN: %clang_cc1 -v -iquote=/usr/include -isysroot /var/empty  -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_IQUOTE_SYSROOT %s
// RUN: %clang_cc1 -v -iquote=/usr/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_IQUOTE_NO_SYSROOT %s
// RUN: %clang_cc1 -v -idirafter=/usr/include -isysroot /var/empty  -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_IDIRAFTER_SYSROOT %s
// RUN: %clang_cc1 -v -idirafter=/usr/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ISYSROOT_IDIRAFTER_NO_SYSROOT %s
// RUN: %clang_cc1 -v -isysroot /var/empty -isystem=/usr/include -iwithsysroot /opt/include -isystem=/usr/local/include -E %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-INTERLEAVE_I_PATHS %s

// RUN: not %clang_cc1 -v -isysroot /var/empty -E %s -o /dev/null -I 2>&1 | FileCheck -check-prefix CHECK-EMPTY_I_PATH %s
// RUN: %clang_cc1 -v -isysroot /var/empty/usr -E %s -o /dev/null -I= 2>&1 | FileCheck -check-prefix CHECK-SYSROOT_I_PATH %s

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

// CHECK-ISYSROOT_ISYSTEM_SYSROOT: ignoring nonexistent directory "/var/empty/usr/include"
// CHECK-ISYSROOT_ISYSTEM_NO_SYSROOT: ignoring nonexistent directory "=/usr/include"

// CHECK-ISYSROOT_IQUOTE_SYSROOT: ignoring nonexistent directory "/var/empty/usr/include"
// CHECK-ISYSROOT_IQUOTE_NO_SYSROOT: ignoring nonexistent directory "=/usr/include"

// CHECK-ISYSROOT_IDIRAFTER_SYSROOT: ignoring nonexistent directory "/var/empty/usr/include"
// CHECK-ISYSROOT_IDIRAFTER_NO_SYSROOT: ignoring nonexistent directory "=/usr/include"

// CHECK-INTERLEAVE_I_PATHS: ignoring nonexistent directory "/var/empty/usr/include"
// CHECK-INTERLEAVE_I_PATHS: ignoring nonexistent directory "/var/empty/opt/include"
// CHECK-INTERLEAVE_I_PATHS: ignoring nonexistent directory "/var/empty/usr/local/include"

// CHECK-EMPTY_I_PATH: argument to '-I' is missing
// CHECK-SYSROOT_I_PATH: ignoring nonexistent directory "/var/empty/usr{{/|\\}}"
