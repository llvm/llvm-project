// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// XFAIL: *
// Our C++0x doesn't currently have specialized destructor name handling,
// since the specification is still in flux.
struct C { 
  typedef int I;
}; 

typedef int I1, I2; 
extern int* p; 
extern int* q; 

void f() {
  p->C::I::~I(); 
  q->I1::~I2();
}

struct A { 
  ~A();
}; 

typedef A AB; 
int main() {
  AB *p; 
  p->AB::~AB();
}
