// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct X1 { // has no implicit default constructor
   X1(int);
};

struct X2  : X1 {
#if __cplusplus <= 199711L
// expected-note@-2 2 {{'X2' declared here}}
#endif

   X2(int);
};

struct X3 : public X2 {
#if __cplusplus <= 199711L
// expected-error@-2 {{implicit default constructor for 'X3' must explicitly initialize the base class 'X2' which does not have a default constructor}}
#else
// expected-note@-4 {{default constructor of 'X3' is implicitly deleted because base class 'X2' has no default constructor}}
#endif
};

X3 x3;
#if __cplusplus <= 199711L
// expected-note@-2 {{first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'X3'}}
#endif

struct X4 {
#if __cplusplus <= 199711L
// expected-error@-2 {{must explicitly initialize the member 'x2'}}
// expected-error@-3 {{must explicitly initialize the reference member 'rx2'}}
#endif

  X2 x2;
#if __cplusplus <= 199711L
  // expected-note@-2 {{member is declared here}}
#else
  // expected-note@-4 {{default constructor of 'X4' is implicitly deleted because field 'x2' has no default constructor}}
#endif

  X2 & rx2;
#if __cplusplus <= 199711L
  // expected-note@-2 {{declared here}}
#endif
};

X4 x4;
#if __cplusplus <= 199711L
// expected-note@-2 {{first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'X4'}}
#endif

struct Y1 { // has no implicit default constructor
   Y1(int);
};

struct Y2  : Y1 { 
   Y2(int);
   Y2();
};

struct Y3 : public Y2 {
};
Y3 y3; 

struct Y4 {
  Y2 y2; 
};

Y4 y4;

// More tests

struct Z1 {
#if __cplusplus <= 199711L
// expected-error@-2 {{must explicitly initialize the reference member 'z'}}
// expected-error@-3 {{must explicitly initialize the const member 'c1'}}
#endif

  int& z;
#if __cplusplus <= 199711L
  // expected-note@-2 {{declared here}}
#else
  // expected-note@-4 {{default constructor of 'Z1' is implicitly deleted because field 'z' of reference type 'int &' would not be initialized}}
#endif

  const int c1;
#if __cplusplus <= 199711L
  // expected-note@-2 {{declared here}}
#endif
  volatile int v1;
};

// Test default initialization which *requires* a constructor call for non-POD.
Z1 z1;
#if __cplusplus <= 199711L
// expected-note@-2 {{first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'Z1'}}
#endif

// Ensure that value initialization doesn't use trivial implicit constructors.
namespace PR7948 {
  // Note that this is also non-POD to ensure we don't just special case PODs.
  struct S { const int x; ~S(); };
  const S arr[2] = { { 42 } };
}

// This is valid
union U {
  const int i;
  float f;
};
U u;
