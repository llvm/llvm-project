// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

int x(1);
int (x2)(1);

void f() {
  int x(1);
  int (x2)(1);
  for (int x(1);;) {}
}

class Y { 
public: explicit Y(float);
};

class X { // expected-note{{candidate constructor (the implicit copy constructor)}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

public:
  explicit X(int); // expected-note{{candidate constructor}}
  X(float, float, float); // expected-note{{candidate constructor}}
  X(float, Y); // expected-note{{candidate constructor}}
};

class Z { // expected-note{{candidate constructor (the implicit copy constructor)}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

public:
  Z(int); // expected-note{{candidate constructor}}
};

void g() {
  X x1(5);
  X x2(1.0, 3, 4.2);
  X x3(1.0, 1.0); // expected-error{{no matching constructor for initialization of 'X'}}
  Y y(1.0);
  X x4(3.14, y);

  Z z; // expected-error{{no matching constructor for initialization of 'Z'}}
}

struct Base {
   operator int*() const; 
};

struct Derived : Base {
   operator int*(); // expected-note {{candidate function}}
};

void foo(const Derived cd, Derived d) {
        int *pi = cd;	// expected-error {{no viable conversion from 'const Derived' to 'int *'}}
        int *ppi = d; 

}
