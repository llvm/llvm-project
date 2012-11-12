// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);

static int count;
static float fcount;

class xpto {
public:
  xpto() : i(count++), f(fcount++) {
    printf("xpto::xpto()\n");
  }
  int i;
  float f;

  ~xpto() {
    printf("xpto::~xpto()\n");
  }
};

int main() {
  xpto array[2][3][4];
  for (int h = 0; h < 2; h++)
   for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
       printf("array[%d][%d][%d] = {%d, %f}\n", 
              h, i, j, array[h][i][j].i, array[h][i][j].f);
}

// CHECK-LP64: callq    __ZN4xptoC1Ev

// CHECK-LP32: calll     L__ZN4xptoC1Ev

