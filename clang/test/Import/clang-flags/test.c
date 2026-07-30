// RUN: clang-import-test -import %S/Inputs/S.c -expression %s -Xcc -DSTRUCT=struct
void expr() {
  STRUCT S MyS;
  void *MyPtr = &MyS;
}
