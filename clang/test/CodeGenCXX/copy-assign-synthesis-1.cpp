// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s

extern "C" int printf(...);

struct B {
  B() : B1(3.14), B2(3.15), auB2(3.16)  {} 
  float B1;
  float B2;
  void pr() {
    printf("B1 = %f B2 = %f auB1 = %f\n", B1, B2, auB1);
  }

  B& operator=(const B& arg) { B1 = arg.B1; B2 = arg.B2; 
                               auB1 = arg.auB1; return *this; }
  union {
    float auB1;
    float auB2;
  };
};

struct M {
  M() : M1(10), M2(11) , auM1(12) {} 
  int M1;
  int M2;
  void pr() {
    printf("M1 = %d M2 = %d auM1 = %d auM2 = %d\n", M1, M2, auM1, auM2);
  }
  union {
    int auM1;
    int auM2;
  };
};

struct N  : B {
  N() : N1(20), N2(21) {} 
  int N1;
  int N2;
  void pr() {
    printf("N1 = %d N2 = %d\n", N1, N2);
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 2; j++)
        printf("arr_b[%d][%d] = %f\n", i,j,arr_b[i][j].B1);
    B::pr();
  }
  N& operator=(const N& arg) {
    N1 = arg.N1; N2 = arg.N2; 
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 2; j++)
        arr_b[i][j] = arg.arr_b[i][j];
    return *this;
  }
  B arr_b[3][2];
};

struct Q  : B {
  Q() : Q1(30), Q2(31) {} 
  int Q1;
  int Q2;
  void pr() {
    printf("Q1 = %d Q2 = %d\n", Q1, Q2);
  }
};


struct X : M , N { 
  X() : d(0.0), d1(1.1), d2(1.2), d3(1.3) {}
  double d;
  double d1;
  double d2;
  double d3;
  void pr() {
    printf("d = %f d1 = %f d2 = %f d3 = %f\n", d, d1,d2,d3);
    M::pr(); N::pr();
    q1.pr(); q2.pr();
  }

 Q q1, q2;
}; 


X srcX; 
X dstX; 
X dstY; 

int main() {
  dstY = dstX = srcX;
  srcX.pr();
  dstX.pr();
  dstY.pr();
}

// CHECK: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %struct.X* @_ZN1XaSERKS_
