// RUN: not clang-import-test -import %S/Inputs/S.c -expression %s 2>&1 | FileCheck %s
// CHECK: error: No such file or directory: {{.*}}Inputs/S.c{{$}}
void expr() {
  struct S MyS;
  void *MyPtr = &MyS;
}
