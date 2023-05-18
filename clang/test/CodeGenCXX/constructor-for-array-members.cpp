// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s

extern "C" int printf(...);

int i = 1234;
float vf = 1.00;

struct S {
  S() : iS(i++), f1(vf++) {printf("S::S()\n");}
  ~S(){printf("S::~S(iS = %d  f1 = %f)\n", iS, f1); }
  int iS;
  float f1;
};

struct M {
  double dM;
  S ARR_S[3];
  void pr() {
    for (int i = 0; i < 3; i++)
     printf("ARR_S[%d].iS = %d ARR_S[%d].f1 = %f\n", i, ARR_S[i].iS, i, ARR_S[i].f1);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
           printf("MULTI_ARR[%d][%d][%d].iS = %d MULTI_ARR[%d][%d][%d].f1 = %f\n", 
                  i,j,k, MULTI_ARR[i][j][k].iS, i,j,k, MULTI_ARR[i][j][k].f1);

  }

 S MULTI_ARR[2][3][4];
};

int main() {
  M m1;
  m1.pr();
}

// CHECK: call void @_ZN1SC1Ev
