// RUN: %clang_cc1 -v -I%S/Inputs -iquote %S/Inputs/SystemHeaderPrefix -isystem %S/Inputs/SystemHeaderPrefix -isystem %S/Inputs/SystemHeaderPrefix %s 2>&1 | FileCheck %s

#include <test.h>

// CHECK: ignoring duplicate directory
// CHECK-SAME: Inputs/SystemHeaderPrefix"{{$}}

// CHECK:      #include "..."
// CHECK-NEXT: {{.*}}Inputs/SystemHeaderPrefix{{$}}
// CHECK-NEXT: #include <...>
// CHECK-NEXT: {{.*}}Inputs{{$}}
// CHECK-NEXT: {{.*}}Inputs/SystemHeaderPrefix{{$}}
