// RUN: clang-import-test -import %S/Inputs/S1.c --import %S/Inputs/S2.c -expression %s
void expr() {
  struct S *MySPtr;
}
