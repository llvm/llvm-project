// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -Wno-microsoft-reference-binding -verify -fms-reference-binding %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -DEXTWARN -fsyntax-only -verify -fms-reference-binding %s

#ifdef EXTWARN

struct A {};
void fARef(A&) {}

void test1() {
  A& a1 = A(); // expected-warning{{binding a user-defined type temporary to a non-const lvalue is a Microsoft extension}}

  fARef(A()); // expected-warning{{binding a user-defined type temporary to a non-const lvalue is a Microsoft extension}}
}

void fARefDoNotWarn(A&) {}
void fARefDoNotWarn(const A&) {}

// expected-note@+2 {{candidate function not viable: 1st argument ('const A') would lose const qualifier}}
// expected-note@+1 {{candidate function not viable: 1st argument ('const A') would lose const qualifier}}
void fARefLoseConstQualifier(A&) {}

void test2() {
  // This should not warn since `fARefDoNotWarn(const A&)` is a better candidate
  fARefDoNotWarn(A());

  const A a;
  fARefLoseConstQualifier(a); // expected-error{{no matching function for call to 'fARefLoseConstQualifier'}}
  fARefLoseConstQualifier(static_cast<const A&&>(a)); // expected-error{{no matching function for call to 'fARefLoseConstQualifier'}}
}

#else

struct A {};
struct B : A {};

typedef A AAlias;

void fADefaultArgRef(A& = A{});
void fBDefaultArgRef(A& = B{});

void fAAliasDefaultArgRef(AAlias& = AAlias{});

B fB();
A fA();

A&& fARvalueRef();
A(&&fARvalueRefArray())[1];

void fADefaultArgRef2(A& = fARvalueRef());

// expected-note@+2 {{candidate function [with T = int] not viable: expects an lvalue for 1st argument}}
template<class T>
void fTRef(T&) {}

void fARef(A&) {}
void fAAliasRef(AAlias&) {}

// expected-note@+2 {{candidate function not viable: expects an lvalue for 1st argument}}
// expected-note@+1 {{candidate function not viable: expects an lvalue for 1st argument}}
void fAVolatileRef(volatile A&) {}

void fIntRef(int&) {} // expected-note{{candidate function not viable: expects an lvalue for 1st argument}}
void fDoubleRef(double&) {} // expected-note{{candidate function not viable: expects an lvalue for 1st argument}}

void fIntConstRef(const int&) {}
void fDoubleConstRef(const double&) {}

void fIntArray(int (&)[1]); // expected-note{{candidate function not viable: expects an lvalue for 1st argument}}
void fIntConstArray(const int (&)[1]);

namespace NS {
  void fARef(A&) {}
  void fAAliasRef(AAlias&) {}

  // expected-note@+2 {{candidate function [with T = int] not viable: expects an lvalue for 1st argument}}
  template<class T>
  void fTRef(T&) {}

  // expected-note@+2 {{passing argument to parameter here}}
  // expected-note@+1 {{passing argument to parameter here}}
  void fAVolatileRef(volatile A&) {}

  void fIntRef(int&) {} // expected-note{{passing argument to parameter here}}
  void fDoubleRef(double&) {} // expected-note{{passing argument to parameter here}}

  void fIntConstRef(const int&) {}
  void fDoubleConstRef(const double&) {}

  A(&&fARvalueRefArray())[1];

  void fIntArray(int (&)[1]); // expected-note{{passing argument to parameter here}}

  void fIntConstArray(const int (&)[1]);
}

void test1() {
  double& rd2 = 2.0; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}
  int& i1 = 0; // expected-error{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}

  fIntRef(0); // expected-error{{no matching function for call to 'fIntRef'}}
  fDoubleRef(0.0); // expected-error{{no matching function for call to 'fDoubleRef'}}

  fTRef(0); // expected-error{{no matching function for call to 'fTRef'}}

  NS::fIntRef(0); // expected-error{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  NS::fDoubleRef(0.0); // expected-error{{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}

  NS::fTRef(0); // expected-error{{no matching function for call to 'fTRef'}}

  int i2 = 2;
  double& rd3 = i2; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}
}

void test2() {
  fIntConstRef(0);
  fDoubleConstRef(0.0);

  NS::fIntConstRef(0);
  NS::fDoubleConstRef(0.0);

  int i = 0;
  const int ci = 0;
  volatile int vi = 0;
  const volatile int cvi = 0;
  bool b = true;

  const volatile int &cvir1 = b ? ci : vi; // expected-error{{volatile lvalue reference to type 'const volatile int' cannot bind to a temporary of type 'int'}}

  volatile int& vir1 = 0; // expected-error{{volatile lvalue reference to type 'volatile int' cannot bind to a temporary of type 'int'}}
  const volatile int& cvir2 = 0; // expected-error{{volatile lvalue reference to type 'const volatile int' cannot bind to a temporary of type 'int'}}
}

void test3() {
  A& a1 = A();
  AAlias& aalias1 = A();

  fARef(A());
  fARef(static_cast<A&&>(a1));

  fTRef(A());
  fTRef(static_cast<A&&>(a1));

  fAAliasRef(A());
  fAAliasRef(static_cast<A&&>(a1));
  fAAliasRef(AAlias());
  fAAliasRef(static_cast<AAlias&&>(a1));

  fAVolatileRef(A()); // expected-error{{no matching function for call to 'fAVolatileRef'}}
  fAVolatileRef(static_cast<A&&>(a1)); // expected-error{{no matching function for call to 'fAVolatileRef'}}

  fARef(B());
  fAAliasRef(B());

  NS::fARef(A());
  NS::fARef(static_cast<A&&>(a1));

  NS::fTRef(A());
  NS::fTRef(static_cast<A&&>(a1));

  NS::fAAliasRef(A());
  NS::fAAliasRef(static_cast<A&&>(a1));
  NS::fAAliasRef(AAlias());
  NS::fAAliasRef(static_cast<AAlias&&>(a1));

  NS::fAVolatileRef(A()); // expected-error{{volatile lvalue reference to type 'volatile A' cannot bind to a temporary of type 'A'}}
  NS::fAVolatileRef(static_cast<A&&>(a1)); // expected-error{{volatile lvalue reference to type 'volatile A' cannot bind to a temporary of type 'A'}}

  NS::fARef(B());
  NS::fAAliasRef(B());

  A& a2 = fA();
  AAlias& aalias2 = fA();

  A& a3 = fARvalueRef();
  AAlias& aalias3 = fARvalueRef();

  const A& rca = fB();
  A& ra = fB();
  AAlias& raalias = fB();
}

void test4() {
  A (&array1)[1] = fARvalueRefArray(); // expected-error{{non-const lvalue reference to type 'A[1]' cannot bind to a temporary of type 'A[1]'}}
  const A (&array2)[1] = fARvalueRefArray();

  A (&array3)[1] = NS::fARvalueRefArray(); // expected-error{{non-const lvalue reference to type 'A[1]' cannot bind to a temporary of type 'A[1]'}}
  const A (&array4)[1] = NS::fARvalueRefArray();

  fIntArray({ 1 }); // expected-error{{no matching function for call to 'fIntArray'}}
  NS::fIntArray({ 1 }); // expected-error{{non-const lvalue reference to type 'int[1]' cannot bind to an initializer list temporary}}

  fIntConstArray({ 1 });
  NS::fIntConstArray({ 1 });
}

void test5() {
  fADefaultArgRef();
  fADefaultArgRef(A());

  fBDefaultArgRef();
  fBDefaultArgRef(B());
  fBDefaultArgRef(A());

  fAAliasDefaultArgRef(A());
  fAAliasDefaultArgRef(B());
  fAAliasDefaultArgRef(AAlias());
}

struct C { operator A() { return A(); } };
struct D { D(int) {} };

// expected-note@+1 {{candidate function not viable: no known conversion from 'C' to 'A &' for 1st argument}}
void fARefConvOperator(A&);

// expected-note@+1 {{candidate function not viable: no known conversion from 'int' to 'D &' for 1st argument}}
void fDRefTemp(D&);

void fAConstRefConvOperator(const A&);
void fDConstRefTemp(const D&);

void test6() {
  fARefConvOperator(C()); // expected-error{{no matching function for call to 'fARefConvOperator'}}
  fDRefTemp(1); // expected-error{{no matching function for call to 'fDRefTemp'}}

  fAConstRefConvOperator(C());
  fDConstRefTemp(1);

  const A& cARef = C();
  A& ARef = C(); // expected-error{{non-const lvalue reference to type 'A' cannot bind to a temporary of type 'C'}}

  const D& cDRef = 1;
  D& DRef = 1; // expected-error{{non-const lvalue reference to type 'D' cannot bind to a temporary of type 'int'}}
}

A& retARef();
struct E { operator A&() { return retARef(); } };

void test7() {
  const A& cARef = E();
  A& ARef = E();
}

struct F { void test(); int i; };

void testFunction() {}
void __vectorcall testVCallFunction() {};

// expected-note@+1 {{candidate function not viable: expects an lvalue for 1st argument}}
void refFuncPtrArg(void (* &)()) {}
void cRefFuncPtrArg(void (* const &)()) {}

void test8() {
  refFuncPtrArg(&testFunction); // expected-error{{no matching function for call to 'refFuncPtrArg'}}
  cRefFuncPtrArg(&testFunction);

  void (* & refFuncPtr1)() = &testFunction; // expected-error{{non-const lvalue reference to type 'void (*)()' cannot bind to a temporary of type 'void (*)()'}}
  void (* const & cRefFuncPtr1)() = &testFunction;

  void (__vectorcall * & refFuncPtr2)() = &testVCallFunction; // expected-error{{non-const lvalue reference to type 'void (*)() __attribute__((vectorcall))' cannot bind to a temporary of type 'void (*)() __attribute__((vectorcall))'}}
  void (__vectorcall * const & cRefFuncPtr2)() = &testVCallFunction;

  void (&refFunc1)() = testFunction;

  void (__vectorcall &refFunc2)() = testVCallFunction;

  void (F::* & refFuncPtr3)() = &F::test; // expected-error{{non-const lvalue reference to type 'void (F::*)()' cannot bind to a temporary of type 'void (F::*)()'}}
  void (F::* const & cRefFuncPtr3)() = &F::test;

  int F::* & refPtr1 = &F::i; // expected-error{{non-const lvalue reference to type 'int F::*' cannot bind to a temporary of type 'int F::*'}}
  int F::* const & cRefPtr1 = &F::i;

  int i;

  int * & refIntPtr1 = &i; // expected-error{{non-const lvalue reference to type 'int *' cannot bind to a temporary of type 'int *'}}
  int * const & cRefIntPtr1 = &i;

  decltype(nullptr) & nullptrRef = nullptr; // expected-error{{non-const lvalue reference to type 'decltype(nullptr)' (aka 'std::nullptr_t') cannot bind to a temporary of type 'std::nullptr_t'}}
  const decltype(nullptr) & nullptrCRef = nullptr;
}

class G {};
union H {};

G fG();
H fH();

enum class I : int {};
enum J { J_ONE = 1, };

void fGRef(G&);
void fHRef(H&);

void test9() {
  G& g1 = fG();
  const G& g2 = fG();

  H& h1 = fH();
  const H& h2 = fH();

  fGRef(fG());
  fHRef(fH());

  I& i1 = I{ 1 }; // expected-error{{non-const lvalue reference to type 'I' cannot bind to a temporary of type 'I'}}
  const I& i2 = I{ 1 };

  J& j1 = J_ONE; // expected-error{{non-const lvalue reference to type 'J' cannot bind to a temporary of type 'J'}}
  const J& j2 = J_ONE;
}

#if defined(__x86_64__)

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));

__m128 fm128();

// expected-note@+1 {{candidate function not viable: expects an lvalue for 1st argument}}
void fm128Ref(__m128&);

// NOTE: Vector types can successfully be bound on msvc since vector types are not built-in types but are unions.
//       We do not implement the msvc model of vector intrinsics so this is expected divergence.
void testm128() {

    fm128Ref(fm128()); // expected-error{{no matching function for call to 'fm128Ref'}}

    __m128& v0 = fm128(); // expected-error{{non-const lvalue reference to type '__m128' (vector of 4 'float' values) cannot bind to a temporary of type '__m128'}}
}

#endif

#endif
