// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s

// CHECK: |-EnumDecl
// CHECK-SAME: Inputs/S.cpp:1:1, line:4:1> line:1:6 E
// CHECK: OpaqueWithType 'long'

void expr() {
  static_assert(E::a + E::b == 3);
  static_assert(sizeof(OpaqueWithType) == sizeof(long));
}
