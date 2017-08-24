// RUN: %clang_cc1 %s -E -o - -I %S/Inputs -isystem %S/Inputs/SystemHeaderPrefix | FileCheck %s
#include <noline.h>
#include <line-directive-in-system.h>

#include "line-directive.h"

// This tests that the line numbers for the current file are correctly outputted
// for the include-file-completed test case.  

// CHECK: # 1 "{{.*}}system-header-line-directive-ms-lineendings.c" 2
// CHECK: # 1 "{{.*}}noline.h" 1 3
// CHECK: foo();
// CHECK: # 3 "{{.*}}system-header-line-directive-ms-lineendings.c" 2
// CHECK: # 1 "{{.*}}line-directive-in-system.h" 1 3
//      The "3" below indicates that "foo.h" is considered a system header.
// CHECK: # 1 "foo.h" 3
// CHECK: foo();
// CHECK: # 4 "{{.*}}system-header-line-directive-ms-lineendings.c" 2
// CHECK: # 1 "{{.*}}line-directive.h" 1
// CHECK: # 10 "foo.h"{{$}}
// CHECK: # 6 "{{.*}}system-header-line-directive-ms-lineendings.c" 2
