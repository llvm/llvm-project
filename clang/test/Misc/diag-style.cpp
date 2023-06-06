// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s

/// empty lines in multi-line diagnostic snippets are preserved.
static_assert(false &&

              true, "");
// CHECK: static assertion failed
// CHECK-NEXT: {{^}}    4 | static_assert(false &&{{$}}
// CHECK-NEXT: {{^}}      |               ^~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    5 | {{$}}
// CHECK-NEXT: {{^}}    6 |               true, "");{{$}}
// CHECK-NEXT: {{^}}      |               ~~~~{{$}}
