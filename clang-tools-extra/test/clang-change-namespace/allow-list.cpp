// RUN: echo "^std::.*$" > %T/allow-list.txt
// RUN: clang-change-namespace -old_namespace "na::nb" -new_namespace "x::y" --file_pattern ".*" --allowed_file %T/allow-list.txt %s -- | sed 's,// CHECK.*,,' | FileCheck %s

#include "Inputs/fake-std.h"

// CHECK: namespace x {
// CHECK-NEXT: namespace y {
namespace na {
namespace nb {
void f() {
  std::STD x1;
  STD x2;
// CHECK: {{^}}  std::STD x1;{{$}}
// CHECK-NEXT: {{^}}  STD x2;{{$}}
}
// CHECK: } // namespace y
// CHECK-NEXT: } // namespace x
}
}
