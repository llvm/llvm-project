// RUN: %clang_cc1 -E -frewrite-includes -include rewrite-includes2.h -I %S/Inputs %s -o - | FileCheck -strict-whitespace %s

// STARTMAIN

// CHECK-NOT: {{^}}#define
// CHECK: included_line2
// CHECK: {{^}}// STARTMAIN{{$}}
