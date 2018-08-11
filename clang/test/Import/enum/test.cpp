// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s

// CHECK: OpaqueWithType 'long'

void expr() {
  static_assert(E::a + E::b == 3);
  static_assert(sizeof(OpaqueWithType) == sizeof(long));
}
