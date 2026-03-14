// RUN: clang-import-test -direct -import %S/Inputs/S.c -expression %s
void expr() {
  struct S MyS;
  MyS.a = 3;
}
