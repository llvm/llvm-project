// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);

class X { // ...
public: 
  X(int) : iX(2), fX(2.3) , name("HELLO\n") {  }

  X(const char* arg, int ix=0) { iX = ix; fX = 6.0; name = arg+ix; }
  X(): iX(100), fX(1.2) {}
  int iX;
  float fX;
  const char *name;
  void pr(void) {
    printf("iX = %d  fX = %f name = %s\n", iX, fX, name);
  }
};

void g(X arg) {
  arg.pr();
}

void f(X arg) {
  X a = 1;        // a = X(1)

  a.pr();

  X b = "Jessie"; //  b=X("Jessie",0)

  b.pr();


  a = 2;          // a = X(2)

  a.pr();
}


int main() {
  X x;
  f(x);
  g(3);           // g(X(3))
}

// CHECK-LP64: callq    __ZN1XC1Ei
// CHECK-LP64: callq    __ZN1XC1EPKci
// CHECK-LP64: callq    __ZN1XC1Ev

// CHECK-LP32: calll     L__ZN1XC1Ei
// CHECK-LP32: calll     L__ZN1XC1EPKci
// CHECK-LP32: calll     L__ZN1XC1Ev
