// RUN: %clang_cc1 -verify -E -frewrite-includes %s -o - | FileCheck -strict-whitespace %s

#include "foobar.h" // expected-error {{'foobar.h' file not found}}
// CHECK: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "foobar.h"
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*}}rewrite-includes-missing.c" 2{{$}}
