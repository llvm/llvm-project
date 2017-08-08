// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, SortIncludes: true}" -lines=1:5 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^#include\ <a>$}}
#include  <a>
// CHECK: {{^#include\ <b>$}}
#include  <b>
#include  <a>
#include  <b>
#include  <b>
{
// CHECK: {{^\ \ int x\ \ ;$}}
  int x  ;
}
