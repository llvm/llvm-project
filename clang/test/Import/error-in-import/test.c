// RUN: not clang-import-test -import %S/Inputs/S.c -expression %s 2>&1 | FileCheck %s
// CHECK: {{.*}}expected unqualified-id{{.*}}
void expr() {
  struct S MyS;
  void *MyPtr = &MyS;
}
