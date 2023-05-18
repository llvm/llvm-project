// RUN: clang-change-namespace -old_namespace "na::nb" -new_namespace "x::y" --file_pattern ".*" %s -- | sed 's,// CHECK.*,,' | FileCheck %s
// CHECK: namespace x {
// CHECK-NEXT: namespace y {
namespace na {
namespace nb {
class A {};
// CHECK: } // namespace y
// CHECK-NEXT: } // namespace x
}
}
