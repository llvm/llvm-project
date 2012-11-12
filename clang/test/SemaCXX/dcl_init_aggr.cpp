// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// C++ [dcl.init.aggr]p2
struct A { 
  int x;
  struct B { 
    int i;
    int j;
  } b; 
} a1 = { 1, { 2, 3 } };

struct NonAggregate {
  NonAggregate();

  int a, b;
};
NonAggregate non_aggregate_test = { 1, 2 }; // expected-error{{non-aggregate type 'NonAggregate' cannot be initialized with an initializer list}}

NonAggregate non_aggregate_test2[2] = { { 1, 2 }, { 3, 4 } }; // expected-error 2 {{non-aggregate type 'NonAggregate' cannot be initialized with an initializer list}}


// C++ [dcl.init.aggr]p3
A a_init = A(); 

// C++ [dcl.init.aggr]p4
int x[] = { 1, 3, 5 };
int x_sizecheck[(sizeof(x) / sizeof(int)) == 3? 1 : -1];
int x2[] = { }; // expected-warning{{zero size arrays are an extension}}

// C++ [dcl.init.aggr]p5
struct StaticMemberTest {
  int i;
  static int s;
  int *j;
} smt = { 1, &smt.i };

// C++ [dcl.init.aggr]p6
char cv[4] = { 'a', 's', 'd', 'f', 0 }; // expected-error{{excess elements in array initializer}}

// C++ [dcl.init.aggr]p7
struct TooFew { int a; char* b; int c; }; 
TooFew too_few = { 1, "asdf" }; // expected-warning{{conversion from string literal to 'char *' is deprecated}}

struct NoDefaultConstructor { // expected-note 3 {{candidate constructor (the implicit copy constructor)}} \
                              // expected-note{{declared here}}
  NoDefaultConstructor(int); // expected-note 3 {{candidate constructor}}
};
struct TooFewError { // expected-error{{implicit default constructor for}}
  int a;
  NoDefaultConstructor nodef; // expected-note{{member is declared here}}
};
TooFewError too_few_okay = { 1, 1 };
TooFewError too_few_error = { 1 }; // expected-error{{no matching constructor}}

TooFewError too_few_okay2[2] = { 1, 1 }; // expected-note{{implicit default constructor for 'TooFewError' first required here}}
TooFewError too_few_error2[2] = { 1 }; // expected-error{{no matching constructor}}

NoDefaultConstructor too_few_error3[3] = { }; // expected-error {{no matching constructor}}

// C++ [dcl.init.aggr]p8
struct Empty { };
struct EmptyTest {
  Empty s;
  int i;
} empty_test = { { }, 3 };

EmptyTest empty_test2 = { 3 }; // expected-error{{initializer for aggregate with no elements requires explicit braces}}

struct NonEmpty { 
  int a;
  Empty empty;
};
struct NonEmptyTest {
  NonEmpty a, b;
} non_empty_test = { { }, { } };

// C++ [dcl.init.aggr]p9
struct HasReference {
  int i;
  int &j; // expected-note{{uninitialized reference member is here}}
};
int global_int;
HasReference r1 = { 1, global_int };
HasReference r2 = { 1 } ; // expected-error{{reference member of type 'int &' uninitialized}}

// C++ [dcl.init.aggr]p10
// Note: the behavior here is identical to C
int xs[2][2] = { 3, 1, 4, 2 };
float y[4][3] = { { 1 }, { 2 }, { 3 }, { 4 } };

// C++ [dcl.init.aggr]p11
// Note: the behavior here is identical to C
float y2[4][3] = { { 1, 3, 5 }, { 2, 4, 6 }, { 3, 5, 7 } };
float same_as_y2[4][3] = { 1, 3, 5, 2, 4, 6, 3, 5, 7 };

// C++ [dcl.init.aggr]p12
struct A2 { 
  int i;
  operator int *();
}; 
struct B2 {
  A2 a1, a2; 
  int *z;
}; 
struct C2 {
  operator A2();
};
struct D2 {
  operator int();
};
A2 a2;
C2 c2; 
D2 d2;
B2 b2 = { 4, a2, a2 };
B2 b2_2 = { 4, d2, 0 };
B2 b2_3 = { c2, a2, a2 };

// C++ [dcl.init.aggr]p15:
union u { int a; char* b; }; // expected-note{{candidate constructor (the implicit copy constructor)}}
u u1 = { 1 }; 
u u2 = u1; 
u u3 = 1; // expected-error{{no viable conversion}}
u u4 = { 0, "asdf" };  // expected-error{{excess elements in union initializer}}
u u5 = { "asdf" }; // expected-error{{cannot initialize a member subobject of type 'int' with an lvalue of type 'const char [5]'}}
