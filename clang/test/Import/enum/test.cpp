// RUN: clang-import-test -import %S/Inputs/S.cpp -expression %s
void expr() {
  static_assert(E::a + E::b == 3);
}
