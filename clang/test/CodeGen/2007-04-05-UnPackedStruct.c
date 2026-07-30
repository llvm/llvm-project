// RUN: %clang_cc1 %s -emit-llvm -o -


enum {
  tA = 0,
  tB = 1
};

struct MyStruct {
  unsigned long A;
  void * B;
};

void bar(void){
struct MyStruct MS = { tB, 0 };
}
