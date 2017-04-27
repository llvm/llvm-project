// RUN: clang-import-test -import %S/Inputs/S.cpp -expression %s
void expr() {
  S MyS;
  int b = MyS.a + MyS.a;
}
