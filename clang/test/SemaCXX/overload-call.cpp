// RUN: %clang_cc1 -triple %itanium_abi_triple -pedantic -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -pedantic -verify -std=c++98 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -pedantic -verify -std=c++11 %s

int* f(int) { return 0; }
float* f(float) { return 0; }
void f();

void test_f(int iv, float fv) {
  float* fp = f(fv);
  int* ip = f(iv);
}

int* g(int, float, int); // expected-note {{candidate function}}
float* g(int, int, int); // expected-note {{candidate function}}
double* g(int, float, float); // expected-note {{candidate function}}
char* g(int, float, ...); // expected-note {{candidate function}}
void g();

void test_g(int iv, float fv) {
  int* ip1 = g(iv, fv, 0);
  float* fp1 = g(iv, iv, 0);
  double* dp1 = g(iv, fv, fv);
  char* cp1 = g(0, 0);
  char* cp2 = g(0, 0, 0, iv, fv);

  double* dp2 = g(0, fv, 1.5); // expected-error {{call to 'g' is ambiguous}}
}

double* h(double f);
int* h(int);

void test_h(float fv, unsigned char cv) {
  double* dp = h(fv);
  int* ip = h(cv);
}

int* i(int);
double* i(long);

void test_i(short sv, int iv, long lv, unsigned char ucv) {
  int* ip1 = i(sv);
  int* ip2 = i(iv);
  int* ip3 = i(ucv);
  double* dp1 = i(lv);
}

int* j(void*);
double* j(bool);

void test_j(int* ip) {
  int* ip1 = j(ip);
}

int* k(char*);
double* k(bool);

void test_k() {
  int* ip1 = k("foo");
#if __cplusplus <= 199711L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
  // expected-error@-4 {{cannot initialize a variable of type 'int *' with an rvalue of type 'double *'}}
#endif

  int* ip2 = k(("foo"));
#if __cplusplus <= 199711L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
  // expected-error@-4 {{cannot initialize a variable of type 'int *' with an rvalue of type 'double *'}}
#endif
  double* dp1 = k(L"foo");
}

int* l(wchar_t*);
double* l(bool);

void test_l() {
  int* ip1 = l(L"foo");
#if __cplusplus <= 199711L
  // expected-warning@-2 {{conversion from string literal to 'wchar_t *' is deprecated}}
#else
  // expected-error@-4 {{cannot initialize a variable of type 'int *' with an rvalue of type 'double *'}}
#endif
  double* dp1 = l("foo");
}

int* m(const char*);
double* m(char*);

void test_m() {
  int* ip = m("foo");
}

int* n(char*);
double* n(void*);
class E;

void test_n(E* e) {
  char ca[7];
  int* ip1 = n(ca);
  int* ip2 = n("foo");
#if __cplusplus <= 199711L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
  // expected-warning@-4 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#endif
  float fa[7];
  double* dp1 = n(fa);

  double* dp2 = n(e);
}

enum PromotesToInt {
  PromotesToIntValue = -1
};

enum PromotesToUnsignedInt {
  PromotesToUnsignedIntValue = __INT_MAX__ * 2U
};

int* o(int);
double* o(unsigned int);
float* o(long);

void test_o() {
  int* ip1 = o(PromotesToIntValue);
  double* dp1 = o(PromotesToUnsignedIntValue);
}

int* p(int);
double* p(double);

void test_p() {
  int* ip = p((short)1);
  double* dp = p(1.0f);
}

struct Bits {
  signed short int_bitfield : 5;
  unsigned int uint_bitfield : 8;
};

int* bitfields(int, int);
float* bitfields(unsigned int, int);

void test_bitfield(Bits bits, int x) {
  int* ip = bitfields(bits.int_bitfield, 0);
  float* fp = bitfields(bits.uint_bitfield, 0u);
}

int* multiparm(long, int, long); // expected-note {{candidate function}}
float* multiparm(int, int, int); // expected-note {{candidate function}}
double* multiparm(int, int, short); // expected-note {{candidate function}}

void test_multiparm(long lv, short sv, int iv) {
  int* ip1 = multiparm(lv, iv, lv);
  int* ip2 = multiparm(lv, sv, lv);
  float* fp1 = multiparm(iv, iv, iv);
  float* fp2 = multiparm(sv, iv, iv);
  double* dp1 = multiparm(sv, sv, sv);
  double* dp2 = multiparm(iv, sv, sv);
  multiparm(sv, sv, lv); // expected-error {{call to 'multiparm' is ambiguous}}
}

// Test overloading based on qualification vs. no qualification
// conversion.
int* quals1(int const * p);
char* quals1(int * p);

int* quals2(int const * const * pp);
char* quals2(int * * pp);

int* quals3(int const * * const * ppp);
char* quals3(int *** ppp);

void test_quals(int * p, int * * pp, int * * * ppp) {
  char* q1 = quals1(p);
  char* q2 = quals2(pp);
  char* q3 = quals3(ppp);
}

// Test overloading based on qualification ranking (C++ 13.3.2)p3.
int* quals_rank1(int const * p);
float* quals_rank1(int const volatile *p);
char* quals_rank1(char*);
double* quals_rank1(const char*);

int* quals_rank2(int const * const * pp);
float* quals_rank2(int * const * pp);

void quals_rank3(int const * const * const volatile * p); // expected-note{{candidate function}}
void quals_rank3(int const * const volatile * const * p); // expected-note{{candidate function}}

void quals_rank3(int const *); // expected-note{{candidate function}}
void quals_rank3(int volatile *); // expected-note{{candidate function}}

void test_quals_ranking(int * p, int volatile *pq, int * * pp, int * * * ppp) {
  int* q1 = quals_rank1(p);
  float* q2 = quals_rank1(pq); 
  double* q3 = quals_rank1("string literal");
  char a[17];
  const char* ap = a;
  char* q4 = quals_rank1(a);
  double* q5 = quals_rank1(ap);

  float* q6 = quals_rank2(pp);

  quals_rank3(ppp); // expected-error {{call to 'quals_rank3' is ambiguous}}

  quals_rank3(p); // expected-error {{call to 'quals_rank3' is ambiguous}}
  quals_rank3(pq);
}

// Test overloading based on derived-to-base conversions
class A { };
class B : public A { };
class C : public B { };
class D : public C { };

int* derived1(A*);
char* derived1(const A*);
float* derived1(void*);

int* derived2(A*);
float* derived2(B*);

int* derived3(A*);
float* derived3(const B*);
char* derived3(C*);

void test_derived(B* b, B const* bc, C* c, const C* cc, void* v, D* d) {
  int* d1 = derived1(b);
  char* d2 = derived1(bc);
  int* d3 = derived1(c);
  char* d4 = derived1(cc);
  float* d5 = derived1(v);

  float* d6 = derived2(b);
  float* d7 = derived2(c);

  char* d8 = derived3(d);
}

void derived4(C*); // expected-note{{candidate function not viable: cannot convert from base class pointer 'A *' to derived class pointer 'C *' for 1st argument}}

void test_base(A* a) {
  derived4(a); // expected-error{{no matching function for call to 'derived4}}
}

// Test overloading of references. 
// (FIXME: tests binding to determine candidate sets, not overload 
//  resolution per se).
int* intref(int&);
float* intref(const int&);

void intref_test() {
  float* ir1 = intref(5);
  float* ir2 = intref(5.5); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 5.5 to 5}}
}

void derived5(C&); // expected-note{{candidate function not viable: cannot bind base class object of type 'A' to derived class reference 'C &' for 1st argument}}

void test_base(A& a) {
  derived5(a); // expected-error{{no matching function for call to 'derived5}}
}

// Test reference binding vs. standard conversions.
int& bind_vs_conv(const double&);
float& bind_vs_conv(int);

void bind_vs_conv_test()
{
  int& i1 = bind_vs_conv(1.0f);
  float& f1 = bind_vs_conv((short)1);
}

// Test that cv-qualifiers get subsumed in the reference binding.
struct X { };
struct Y { };
struct Z : X, Y { };

int& cvqual_subsume(X&); // expected-note{{candidate function}}
float& cvqual_subsume(const Y&); // expected-note{{candidate function}}

int& cvqual_subsume2(X&); // expected-note{{candidate function}}
float& cvqual_subsume2(volatile Y&); // expected-note{{candidate function}}

void cvqual_subsume_test(Z z) {
  cvqual_subsume(z); // expected-error{{call to 'cvqual_subsume' is ambiguous}}
  cvqual_subsume2(z); // expected-error{{call to 'cvqual_subsume2' is ambiguous}}
}

// Test overloading with cv-qualification differences in reference
// binding.
int& cvqual_diff(X&);
float& cvqual_diff(const X&);

void cvqual_diff_test(X x, Z z) {
  int& i1 = cvqual_diff(x);
  int& i2 = cvqual_diff(z);
}

// Test overloading with derived-to-base differences in reference
// binding.
struct Z2 : Z { };

int& db_rebind(X&);
long& db_rebind(Y&);
float& db_rebind(Z&);

void db_rebind_test(Z2 z2) {
  float& f1 = db_rebind(z2);
}

class string { };
class opt : public string { };

struct SR {
  SR(const string&);
};

void f(SR) { }

void g(opt o) {
  f(o);
}


namespace PR5756 {
  int &a(void*, int);
  float &a(void*, float);
  void b() { 
    int &ir = a(0,0);
    (void)ir;
  }
}

// Tests the exact text used to note the candidates
namespace test1 {
template <class T>
void foo(T t, unsigned N);                        // expected-note {{candidate function template not viable: no known conversion from 'const char [6]' to 'unsigned int' for 2nd argument}}
void foo(int n, char N);                          // expected-note {{candidate function not viable: no known conversion from 'const char [6]' to 'char' for 2nd argument}}
void foo(int n, const char *s, int t);            // expected-note {{candidate function not viable: requires 3 arguments, but 2 were provided}}
void foo(int n, const char *s, int t, ...);       // expected-note {{candidate function not viable: requires at least 3 arguments, but 2 were provided}}
void foo(int n, const char *s, int t, int u = 0); // expected-note {{candidate function not viable: requires at least 3 arguments, but 2 were provided}}

// PR 11857
void foo(int n);                // expected-note {{candidate function not viable: requires single argument 'n', but 2 arguments were provided}}
void foo(unsigned n = 10);      // expected-note {{candidate function not viable: allows at most single argument 'n', but 2 arguments were provided}}
void bar(int n, int u = 0);     // expected-note {{candidate function not viable: requires at least argument 'n', but no arguments were provided}}
void baz(int n = 0, int u = 0); // expected-note {{candidate function not viable: requires at most 2 arguments, but 3 were provided}}

void test() {
  foo(4, "hello"); //expected-error {{no matching function for call to 'foo'}}
  bar();           //expected-error {{no matching function for call to 'bar'}}
  baz(3, 4, 5);    // expected-error {{no matching function for call to 'baz'}}
  }
}

// PR 6014
namespace test2 {
  struct QFixed {
    QFixed(int i);
    QFixed(long i);
  };

  bool operator==(const QFixed &f, int i);

  class qrgb666 {
    inline operator unsigned int () const;

    inline bool operator==(const qrgb666 &v) const;
    inline bool operator!=(const qrgb666 &v) const { return !(*this == v); }
  };
}

// PR 6117
namespace IncompleteConversion {
  struct Complete {};
  struct Incomplete;

  void completeFunction(Complete *); // expected-note 2 {{cannot convert argument of incomplete type}}
  void completeFunction(Complete &); // expected-note 2 {{cannot convert argument of incomplete type}}
  
  void testTypeConversion(Incomplete *P) {
    completeFunction(P); // expected-error {{no matching function for call to 'completeFunction'}}
    completeFunction(*P); // expected-error {{no matching function for call to 'completeFunction'}}
  }
  
  void incompletePointerFunction(Incomplete *); // expected-note {{candidate function not viable: cannot convert argument of incomplete type 'IncompleteConversion::Incomplete' to 'IncompleteConversion::Incomplete *' for 1st argument; take the address of the argument with &}}
  void incompleteReferenceFunction(Incomplete &); // expected-note {{candidate function not viable: cannot convert argument of incomplete type 'IncompleteConversion::Incomplete *' to 'IncompleteConversion::Incomplete &' for 1st argument; dereference the argument with *}}
  
  void testPointerReferenceConversion(Incomplete &reference, Incomplete *pointer) {
    incompletePointerFunction(reference); // expected-error {{no matching function for call to 'incompletePointerFunction'}}
    incompleteReferenceFunction(pointer); // expected-error {{no matching function for call to 'incompleteReferenceFunction'}}
  }
}

namespace DerivedToBaseVsVoid {
  struct A { };
  struct B : A { };
  
  float &f(void *);
  int &f(const A*);
  
  void g(B *b) {
    int &ir = f(b);
  }
}

// PR 6398 + PR 6421
namespace test4 {
  class A;
  class B {
    static void foo(); // expected-note {{not viable}}
    static void foo(int*); // expected-note {{not viable}}
    static void foo(long*); // expected-note {{not viable}}

    void bar(A *a) { 
      foo(a); // expected-error {{no matching function for call}}
    }
  };
}

namespace DerivedToBase {
  struct A { };
  struct B : A { };
  struct C : B { };
  
  int &f0(const A&);
  float &f0(B);
  
  void g() {
    float &fr = f0(C());
  }
}

namespace PR6483 {
  struct X0 {
    operator const unsigned int & () const;
  };

  struct X1 {
    operator unsigned int & () const;
  };

  void f0(const bool &);
  void f1(bool &); // expected-note 2{{not viable}}

  void g(X0 x0, X1 x1) {
    f0(x0);
    f1(x0); // expected-error{{no matching function for call}}
    f0(x1);
    f1(x1); // expected-error{{no matching function for call}}
  }  
}

namespace PR6078 {
  struct A {
    A(short); // expected-note{{candidate constructor}}
    A(long); // expected-note{{candidate constructor}}
  };
  struct S {
    typedef void ft(A);
    operator ft*();
  };

  void f() {
    S()(0); // expected-error{{conversion from 'int' to 'PR6078::A' is ambiguous}}
  }
}

namespace PR6177 {
  struct String { String(char const*); };

  void f(bool const volatile&);
  int &f(String);

  void g() { int &r = f(""); }
}

namespace PR7095 {
  struct X { };

  struct Y {
    operator const X*();

  private:
    operator X*();
  };

  void f(const X *);
  void g(Y y) { f(y); }
}

namespace PR7224 {
  class A {};
  class B : public A {};

  int &foo(A *const d);
  float &foo(const A *const d);

  void bar()
  {
    B *const d = 0;
    B const *const d2 = 0;
    int &ir = foo(d);
    float &fr = foo(d2);
  }
}

namespace NontrivialSubsequence {
  struct X0;

  class A {
    operator X0 *();
  public:
    operator const X0 *();
  };
 
  A a;
  void foo( void const * );

  void g() {
    foo(a);
  }
}

// rdar://rdar8499524
namespace rdar8499524 {
  struct W {};
  struct S {
      S(...);
  };

  void g(const S&);
  void f() {
    g(W());
  }
}

namespace rdar9173984 {
  template <typename T, unsigned long N> int &f(const T (&)[N]);
  template <typename T> float &f(const T *);

  void test() {
    int arr[2] = {0, 0};
    int *arrp = arr;
    int &ir = f(arr);
    float &fr = f(arrp);
  }
}

namespace PR9507 {
  void f(int * const&); // expected-note{{candidate function}}
  void f(int const(&)[1]); // expected-note{{candidate function}}
 
  int main() {
    int n[1];
    f(n); // expected-error{{call to 'f' is ambiguous}}
  }
}

namespace rdar9803316 {
  void foo(float);
  int &foo(int);

  void bar() {
    int &ir = (&foo)(0);
  }
}

namespace IncompleteArg {
  // Ensure that overload resolution attempts to complete argument types when
  // performing ADL.
  template<typename T> struct S {
    friend int f(const S&);
  };
  extern S<int> s;
  int k = f(s);

  template<typename T> struct Op {
    friend bool operator==(const Op &, const Op &);
  };
  extern Op<char> op;
  bool b = op == op;

  // ... and not in other cases! Nothing here requires U<int()> to be complete.
  // (Note that instantiating U<int()> will fail.)
  template<typename T> struct U {
    T t;
  };
  struct Consumer {
    template<typename T>
    int operator()(const U<T> &);
  };
  template<typename T> U<T> &make();
  Consumer c;
  int n = sizeof(c(make<int()>()));
}

namespace PR12142 {
  void fun(int (*x)[10]); // expected-note{{candidate function not viable: 1st argument ('const int (*)[10]') would lose const qualifier}}
  void g() { fun((const int(*)[10])0); } // expected-error{{no matching function for call to 'fun'}}
}

// DR1152: Take 'volatile' into account when handling reference bindings in
//         overload resolution.
namespace PR12931 {
  void f(const int &, ...);
  void f(const volatile int &, int);
  void g() { f(0, 0); }
}

void test5() {
  struct {
    typedef void F1(int);
    typedef void F2(double);
    operator F1*();  // expected-note{{conversion candidate}}
    operator F2*();  // expected-note{{conversion candidate}}
  } callable;
  callable();  // expected-error{{no matching function for call}}
}

namespace PR20218 {
  void f(void (*const &)()); // expected-note 2{{candidate}}
  void f(void (&&)()) = delete; // expected-note 2{{candidate}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{rvalue references are a C++11 extension}}
  // expected-warning@-3 {{deleted function definitions are a C++11 extension}}
#endif
  void g(void (&&)()) = delete; // expected-note 2{{candidate}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{rvalue references are a C++11 extension}}
  // expected-warning@-3 {{deleted function definitions are a C++11 extension}}
#endif
  void g(void (*const &)()); // expected-note 2{{candidate}}

  void x();
  typedef void (&fr)();
  struct Y { operator fr(); } y;

  void h() {
    f(x); // expected-error {{ambiguous}}
    g(x); // expected-error {{ambiguous}}
    f(y); // expected-error {{ambiguous}}
    g(y); // expected-error {{ambiguous}}
  }
}

namespace StringLiteralToCharAmbiguity {
  void f(char *, int);
  void f(const char *, unsigned);
  void g() { f("foo", 0); }
#if __cplusplus <= 199711L
  // expected-error@-2 {{call to 'f' is ambiguous}}
  // expected-note@-5 {{candidate function}}
  // expected-note@-5 {{candidate function}}
#endif
}

namespace ProduceNotesAfterSFINAEFailure {
  struct A {
    template<typename T, typename U = typename T::x> A(T); // expected-warning 0-1{{extension}}
  };
  void f(void*, A); // expected-note {{candidate function not viable}}
  void g() { f(1, 2); } // expected-error {{no matching function}}
}
