// RUN: %clang_cc1 -no-opaque-pointers -std=c++1z -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-passes -o - %s | FileCheck %s

// std::byte should be considered equivalent to char for aliasing.

namespace std {
enum byte : unsigned char {};
}

// CHECK-LABEL: define{{.*}} void @test0(
extern "C" void test0(std::byte *sb, int *i) {
  // CHECK: store i8 0, i8* %{{.*}} !tbaa [[TAG_CHAR:!.*]]
  *sb = std::byte{0};

  // CHECK: store i32 1, i32* %{{.*}} !tbaa [[TAG_INT:!.*]]
  *i = 1;
}

enum byte : unsigned char {};
namespace my {
enum byte : unsigned char {};
namespace std {
enum byte : unsigned char {};
} // namespace std
} // namespace my

// Make sure we don't get confused with other enums named 'byte'.

// CHECK-LABEL: define{{.*}} void @test1(
extern "C" void test1(::byte *b, ::my::byte *mb, ::my::std::byte *msb) {
  *b = ::byte{0};
  *mb = ::my::byte{0};
  *msb = ::my::std::byte{0};
  // CHECK-NOT: store i8 0, i8* %{{.*}} !tbaa [[TAG_CHAR]]
}

// CHECK:  !"any pointer", [[TYPE_CHAR:!.*]],
// CHECK: [[TYPE_CHAR]] = !{!"omnipotent char", [[TAG_CXX_TBAA:!.*]],
// CHECK: [[TAG_CXX_TBAA]] = !{!"Simple C++ TBAA"}
// CHECK: [[TAG_CHAR]] = !{[[TYPE_CHAR:!.*]], [[TYPE_CHAR]], i64 0}
// CHECK: [[TAG_INT]] = !{[[TYPE_INT:!.*]], [[TYPE_INT]], i64 0}
// CHECK: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR]]
