// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s

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

// CHECK: call void @_ZN4xptoC1Ev
