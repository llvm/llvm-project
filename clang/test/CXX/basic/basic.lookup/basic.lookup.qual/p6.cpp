// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

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
