// Setup:
// RUN: rm -rf %t
// RUN: mkdir -p %t/usr/include
// RUN: echo '// empty' > %t/usr/include/sys_header.h
// RUN: cp %s %t.h
//
// Precompile
// RUN: %clang_cc1 -isystem %t/usr/include -x objective-c-header -emit-pch -o %t.pch %t.h

// Verify successfully
// RUN: %clang_cc1 -isystem %t/usr/include -verify-pch %t.pch

// Incompatible lang options ignored
// RUN: %clang_cc1 -isystem %t/usr/include -x objective-c -fno-builtin -verify-pch %t.pch

// Stale dependency
// RUN: echo ' ' >> %t.h
// RUN: not %clang_cc1 -isystem %t/usr/include -verify-pch %t.pch 2> %t.log.2
// RUN: FileCheck -check-prefix=CHECK-STALE-DEP %s < %t.log.2
// CHECK-STALE-DEP: file '{{.*}}.h' has been modified since the precompiled header '{{.*}}.pch' was built

// Stale dependency in system header
// RUN: %clang_cc1 -isystem %t/usr/include -x objective-c-header -emit-pch -o %t.pch %t.h
// RUN: %clang_cc1 -isystem %t/usr/include -verify-pch %t.pch
// RUN: echo ' ' >> %t/usr/include/sys_header.h
// RUN: not %clang_cc1 -isystem %t/usr/include -verify-pch %t.pch 2> %t.log.3
// RUN: FileCheck -check-prefix=CHECK-STALE-SYS-H %s < %t.log.3
// CHECK-STALE-SYS-H: file '{{.*}}sys_header.h' has been modified since the precompiled header '{{.*}}.pch' was built

#include <sys_header.h>
