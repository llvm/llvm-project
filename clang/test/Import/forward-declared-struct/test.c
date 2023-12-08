// RUN: clang-import-test -import %S/Inputs/S1.c --import %S/Inputs/S2.c --import %S/Inputs/S3.c -expression %s
void expr() {
  struct S MyS;
  MyS.a = 3;
}
