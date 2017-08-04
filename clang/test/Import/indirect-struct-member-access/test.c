// RUN: clang-import-test -import %S/Inputs/S.c -expression %s
void expr(struct S *MyS) {
  MyS->a = 3;
}
