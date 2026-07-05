// RUN: %clang_cc1 %s -emit-llvm -o -

#pragma pack(push, 2)

enum {
  tA = 0,
  tB = 1
};

struct MyStruct {
  unsigned long A;
  char C;
  void * B;
};

void bar(void){
struct MyStruct MS = { tB, 0 };
}
