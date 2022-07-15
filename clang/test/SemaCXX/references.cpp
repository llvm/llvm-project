// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
int g(int);

void f() {
  int i;
  int &r = i;
  r = 1;
  int *p = &r;
  int &rr = r;
  int (&rg)(int) = g;
  rg(i);
  int a[3];
  int (&ra)[3] = a;
  ra[1] = i;
  int *Q;
  int *& P = Q;
  P[1] = 1;
}

typedef int t[1];
void test2() {
    t a;
    t& b = a;


    int c[3];
    int (&rc)[3] = c;
}

// C++ [dcl.init.ref]p5b1
struct A { };
struct B : A { } b;

void test3() {
  double d = 2.0;
  double& rd = d; // rd refers to d
  const double& rcd = d; // rcd refers to d

  A& ra = b; // ra refers to A subobject in b
  const A& rca = b; // rca refers to A subobject in b
}

B fB();

// C++ [dcl.init.ref]p5b2
void test4() {
  double& rd2 = 2.0; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}
  int i = 2;
  double& rd3 = i; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}

  const A& rca = fB();
}

void test5() {
  //  const double& rcd2 = 2; // rcd2 refers to temporary with value 2.0
  const volatile int cvi = 1;
  const int& r = cvi; // expected-error{{binding reference of type 'const int' to value of type 'const volatile int' drops 'volatile' qualifier}}

#if __cplusplus >= 201103L
  const int& r2{cvi}; // expected-error{{binding reference of type 'const int' to value of type 'const volatile int' drops 'volatile' qualifier}}

  const int a = 2;
  int& r3{a}; // expected-error{{binding reference of type 'int' to value of type 'const int' drops 'const' qualifier}}

  const int&& r4{a}; // expected-error{{rvalue reference to type 'const int' cannot bind to lvalue of type 'const int'}}

  void func();
  void func(int);
  int &ptr1 = {func}; // expected-error{{address of overloaded function 'func' does not match required type 'int'}}
  int &&ptr2{func}; // expected-error{{address of overloaded function 'func' does not match required type 'int'}}
  // expected-note@-4{{candidate function}}
  // expected-note@-4{{candidate function}}
  // expected-note@-6{{candidate function}}
  // expected-note@-6{{candidate function}}
#endif
}

// C++ [dcl.init.ref]p3
int& test6(int& x) {
  int& yo; // expected-error{{declaration of reference variable 'yo' requires an initializer}}

  return x;
}
int& not_initialized_error; // expected-error{{declaration of reference variable 'not_initialized_error' requires an initializer}}
extern int& not_initialized_okay;

class Test6 { // expected-warning{{class 'Test6' does not declare any constructor to initialize its non-modifiable members}}
  int& okay; // expected-note{{reference member 'okay' will never be initialized}}
};

struct C : B, A { }; // expected-warning {{direct base 'A' is inaccessible due to ambiguity:\n    struct C -> struct B -> struct A\nstruct C -> struct A}}

void test7(C& c) {
  A& a1 = c; // expected-error {{ambiguous conversion from derived class 'C' to base class 'A':}}
}

// C++ [dcl.ref]p1, C++ [dcl.ref]p4
void test8(int& const,// expected-error{{'const' qualifier may not be applied to a reference}}
           
           void&,     // expected-error{{cannot form a reference to 'void'}}
           int& &)    // expected-error{{type name declared as a reference to a reference}}
{
  typedef int& intref;
  typedef intref& intrefref; // C++ DR 106: reference collapsing

  typedef intref const intref_c; // expected-warning {{'const' qualifier on reference type 'intref' (aka 'int &') has no effect}}
  typedef intref_c intref; // ok, same type

  typedef intref volatile intref; // expected-warning {{'volatile' qualifier on reference type 'intref' (aka 'int &') has no effect}}
  typedef intref _Atomic intref; // expected-warning {{'_Atomic' qualifier on reference type 'intref' (aka 'int &') has no effect}}

  void restrict_ref(__restrict intref); // ok
  void restrict_ref(int &__restrict); // ok
}

namespace var_template {
#if __cplusplus >= 201402L
int i;
template <typename> int &ref = i; // ok
template <> int &ref<float>;      // expected-error {{declaration of reference variable 'ref<float>' requires an initializer}}
#endif
} // namespace var_template

template<typename T> int const_param(const T) {}
int const_ref_param = const_param<int&>(const_ref_param); // no-warning


class string {
  char *Data;
  unsigned Length;
public:
  string(); 
  ~string();
};

string getInput();

void test9() {
  string &s = getInput(); // expected-error{{lvalue reference}}
}

void test10() {
  __attribute((vector_size(16))) typedef int vec4;
  typedef __attribute__(( ext_vector_type(4) )) int ext_vec4;
  
  vec4 v;
  int &a = v[0]; // expected-error{{non-const reference cannot bind to vector element}}
  const int &b = v[0];
  
  ext_vec4 ev;
  int &c = ev.x; // expected-error{{non-const reference cannot bind to vector element}}
  const int &d = ev.x;
}

namespace PR7149 {
  template<typename T> struct X0
  {
    T& first;
    X0(T& p1) : first(p1) { }
  };


  void f()
  {
    int p1[1];
    X0< const int[1]> c(p1);
  }
}

namespace PR8608 {
  bool& f(unsigned char& c) { return (bool&)c; }
}

// The following crashed trying to recursively evaluate the LValue.
const int &do_not_crash = do_not_crash; // expected-warning{{reference 'do_not_crash' is not yet bound to a value when used within its own initialization}}

namespace ExplicitRefInit {
  // This is invalid: we can't copy-initialize an 'A' temporary using an
  // explicit constructor.
  struct A { explicit A(int); };
  const A &a(0); // expected-error {{reference to type 'const ExplicitRefInit::A' could not bind to an rvalue of type 'int'}}
}

namespace RefCollapseTypePrinting {
  template<typename T> void add_lref() {
    using X = int(T); // expected-note 4{{previous}}
    using X = const volatile T&;
    // expected-error@-1 {{'int &' vs 'int (int &)'}}
    // expected-error@-2 {{'int &' vs 'int (int &&)'}}
    // expected-error@-3 {{'const int &' vs 'int (const int &)'}}
    // expected-error@-4 {{'const int &' vs 'int (const int &&)'}}
  }
  template void add_lref<int&>(); // expected-note {{instantiation of}}
  template void add_lref<int&&>(); // expected-note {{instantiation of}}
  template void add_lref<const int&>(); // expected-note {{instantiation of}}
  template void add_lref<const int&&>(); // expected-note {{instantiation of}}

  template<typename T> void add_rref() {
    using X = int(T); // expected-note 4{{previous}}
    using X = const volatile T&&;
    // expected-error@-1 {{'int &' vs 'int (int &)'}}
    // expected-error@-2 {{'int &&' vs 'int (int &&)'}}
    // expected-error@-3 {{'const int &' vs 'int (const int &)'}}
    // expected-error@-4 {{'const int &&' vs 'int (const int &&)'}}
  }
  template void add_rref<int&>(); // expected-note {{instantiation of}}
  template void add_rref<int&&>(); // expected-note {{instantiation of}}
  template void add_rref<const int&>(); // expected-note {{instantiation of}}
  template void add_rref<const int&&>(); // expected-note {{instantiation of}}
}

namespace PR45521 {
  struct a { template<class b> a(const b * const&); };
  int *d;
  const a &r = d;
}
