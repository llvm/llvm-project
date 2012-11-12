// RUN: %clang_cc1 -emit-llvm < %s | grep i1 | count 1
// All of these should uses the memory representation of _Bool
struct teststruct1 {_Bool a, b;} test1;
_Bool* test2;
_Bool test3[10];
_Bool (*test4)[];
void f(int x) {
  _Bool test5;
  _Bool test6[x];
}
