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


/// #line pragmas are respected
void printf(const char *a, ...) __attribute__((__format__(__printf__, 1, 2)));
#line 10
void f(int x) {
  printf("%f",
         x);
}
// CHECK: 12:10: warning: format specifies type
// CHECK-NEXT: {{^}}   11 |
// CHECK-NEXT: {{^}}      |
// CHECK-NEXT: {{^}}      |
// CHECK-NEXT: {{^}}   12 |

#line 10
int func(
  int a, int b,
  int&
  r);

void test() {
  func(3, 4, 5);
}
// CHECK: 10:5: note: candidate function not viable
// CHECK-NEXT: {{^}}   10 |
// CHECK-NEXT: {{^}}      |
// CHECK-NEXT: {{^}}   11 |
// CHECK-NEXT: {{^}}   12 |
// CHECK-NEXT: {{^}}      |
// CHECK-NEXT: {{^}}   13 |
// CHECK-NEXT: {{^}}      |

