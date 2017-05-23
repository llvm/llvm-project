// RUN: %clang_cc1 -Wall %s -isystem %S/Inputs/SystemHeaderPrefix -verify
// RUN: %clang_cc1 %s -E -o - -isystem %S/Inputs/SystemHeaderPrefix | FileCheck %s
#include <noline.h>
#include <line.h>

// This tests that "#line" directives in system headers preserve system
// header-ness just like GNU line markers that don't have filenames.  This was
// PR30752.

// expected-no-diagnostics

// CHECK: # {{[0-9]+}} "{{.*}}system-header-line-directive.c" 2
// CHECK: # 1 "{{.*}}noline.h" 1 3
// CHECK: foo();
// CHECK: # 4 "{{.*}}system-header-line-directive.c" 2
// CHECK: # 1 "{{.*}}line.h" 1 3
//      The "3" below indicates that "foo.h" is considered a system header.
// CHECK: # 1 "foo.h" 3
// CHECK: foo();
// CHECK: # {{[0-9]+}} "{{.*}}system-header-line-directive.c" 2
