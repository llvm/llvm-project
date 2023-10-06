// RUN: not %clang_cc1 -verify -E -frewrite-includes -include %S/Inputs/rewrite-includes2.h %s -o - | FileCheck -strict-whitespace %s
main_file_line
// CHECK: {{^}}# 1 "<built-in>"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes2.h" 1{{$}}
// CHECK-NEXT: {{^}}int included_line2;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 1 "<built-in>" 2{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*}}rewrite-includes-cli-include.c"{{$}}
// CHECK-NEXT: FileCheck
// CHECK-NEXT: {{^}}main_file_line{{$}}
