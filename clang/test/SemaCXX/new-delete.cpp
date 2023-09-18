// RUN: %clang_cc1 -fsyntax-only -verify %s -triple=i686-pc-linux-gnu -Wno-new-returns-null -std=c++98
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple=i686-pc-linux-gnu -Wno-new-returns-null -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple=i686-pc-linux-gnu -Wno-new-returns-null -std=c++14
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 %s -triple=i686-pc-linux-gnu -Wno-new-returns-null %std_cxx17-

// FIXME Location is (frontend)
// cxx17-note@*:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}

#include <stddef.h>

#if __cplusplus >= 201103L
// expected-note@+2 {{candidate constructor (the implicit move constructor) not viable}}
#endif
struct S // expected-note {{candidate}}
{
  S(int, int, double); // expected-note {{candidate}}
  S(double, int); // expected-note 2 {{candidate}}
  S(float, int); // expected-note 2 {{candidate}}
};
struct T; // expected-note{{forward declaration of 'T'}}
struct U
{
  // A special new, to verify that the global version isn't used.
  void* operator new(size_t, S*); // expected-note {{candidate}}
};
struct V : U
{
};

inline void operator delete(void *); // expected-warning {{replacement function 'operator delete' cannot be declared 'inline'}}

__attribute__((used))
inline void *operator new(size_t) { // no warning, due to __attribute__((used))
  return 0; // expected-warning {{null returned from function that requires a non-null return value}}
}

// PR5823
void* operator new(const size_t); // expected-note {{candidate}}
void* operator new(size_t, int*); // expected-note 2{{candidate}}
void* operator new(size_t, float*); // expected-note 2{{candidate}}
void* operator new(size_t, S); // expected-note {{candidate}}

struct foo { };

void good_news()
{
  int *pi = new int;
  float *pf = new (pi) float();
  pi = new int(1);
  pi = new int('c');
  const int *pci = new const int();
  S *ps = new S(1, 2, 3.4);
  ps = new (pf) (S)(1, 2, 3.4);
  S *(*paps)[2] = new S*[*pi][2];
  typedef int ia4[4];
  ia4 *pai = new (int[3][4]);
  pi = ::new int;
  U *pu = new (ps) U;
  V *pv = new (ps) V;

  pi = new (S(1.0f, 2)) int;

  (void)new int[true];

  // PR7147
  typedef int a[2];
  foo* f1 = new foo;
  foo* f2 = new foo[2];
  typedef foo x[2];
  typedef foo y[2][2];
  x* f3 = new y;

#if __cplusplus >= 201103L
  (void)new int[]{};
  (void)new int[]{1, 2, 3};
  (void)new char[]{"hello"};
#endif
}

struct abstract {
  virtual ~abstract() = 0;
};

void bad_news(int *ip)
{
  int i = 1; // expected-note 2{{here}}
  (void)new; // expected-error {{expected a type}}
  (void)new 4; // expected-error {{expected a type}}
  (void)new () int; // expected-error {{expected expression}}
  (void)new int[1.1];
#if __cplusplus <= 199711L
  // expected-error@-2 {{array size expression must have integral or enumeration type, not 'double'}}
#elif __cplusplus <= 201103L
  // expected-error@-4 {{array size expression must have integral or unscoped enumeration type, not 'double'}}
#else
  // expected-warning@-6 {{implicit conversion from 'double' to 'unsigned int' changes value from 1.1 to 1}}
#endif

  (void)new int[1][i];  // expected-note {{read of non-const variable 'i' is not allowed in a constant expression}}
  (void)new (int[1][i]); // expected-note {{read of non-const variable 'i' is not allowed in a constant expression}}
#if __cplusplus <= 201103L
  // expected-error@-3 {{only the first dimension}}
  // expected-error@-3 {{only the first dimension}}
#else
  // expected-error@-6 {{array size is not a constant expression}}
  // expected-error@-6 {{array size is not a constant expression}}
#endif
  (void)new (int[i]); // expected-warning {{when type is in parentheses}}
  (void)new int(*(S*)0); // expected-error {{no viable conversion from 'S' to 'int'}}
  (void)new int(1, 2); // expected-error {{excess elements in scalar initializer}}
  (void)new S(1); // expected-error {{no matching constructor}}
  (void)new S(1, 1); // expected-error {{call to constructor of 'S' is ambiguous}}
  (void)new const int; // expected-error {{default initialization of an object of const type 'const int'}}
  (void)new float*(ip); // expected-error {{cannot initialize a new value of type 'float *' with an lvalue of type 'int *'}}
  // Undefined, but clang should reject it directly.
  (void)new int[-1];
#if __cplusplus <= 201103L
  // expected-error@-2 {{array size is negative}}
#else
  // expected-error@-4 {{array is too large}}
#endif
  (void)new int[2000000000]; // expected-error {{array is too large}}
  (void)new int[*(S*)0];
#if __cplusplus <= 199711L
  // expected-error@-2 {{array size expression must have integral or enumeration type, not 'S'}}
#elif __cplusplus <= 201103L
  // expected-error@-4 {{array size expression must have integral or unscoped enumeration type, not 'S'}}
#else
  // expected-error@-6 {{converting 'S' to incompatible type}}
#endif

  (void)::S::new int; // expected-error {{expected unqualified-id}}
  (void)new (0, 0) int; // expected-error {{no matching function for call to 'operator new'}}
  (void)new (0L) int; // expected-error {{call to 'operator new' is ambiguous}}
  // This must fail, because the member version shouldn't be found.
  (void)::new ((S*)0) U; // expected-error {{no matching 'operator new' function for non-allocating placement new expression; include <new>}}
  // This must fail, because any member version hides all global versions.
  (void)new U; // expected-error {{no matching function for call to 'operator new'}}
  (void)new (int[]); // expected-error {{array size must be specified in new expression with no initializer}}
  (void)new int&; // expected-error {{cannot allocate reference type 'int &' with new}}
  (void)new int[]; // expected-error {{array size must be specified in new expression with no initializer}}
  (void)new int[](); // expected-error {{cannot determine allocated array size from initializer}}
  // FIXME: This is a terrible diagnostic.
#if __cplusplus < 201103L
  (void)new int[]{}; // expected-error {{array size must be specified in new expression with no initializer}}
#endif
}

void no_matching_placement_new() {
  struct X { int n; };
  __attribute__((aligned(__alignof(X)))) unsigned char buffer[sizeof(X)];
  (void)new(buffer) X; // expected-error {{no matching 'operator new' function for non-allocating placement new expression; include <new>}}
  (void)new(+buffer) X; // expected-error {{no matching 'operator new' function for non-allocating placement new expression; include <new>}}
  (void)new(&buffer) X; // expected-error {{no matching 'operator new' function for non-allocating placement new expression; include <new>}}
}

void good_deletes()
{
  delete (int*)0;
  delete [](int*)0;
  delete (S*)0;
  ::delete (int*)0;
}

void bad_deletes()
{
  delete 0; // expected-error {{cannot delete expression of type 'int'}}
  delete [0] (int*)0;
#if __cplusplus <= 199711L
  // expected-error@-2 {{expected expression}}
#else
  // expected-error@-4 {{expected variable name or 'this' in lambda capture list}}
#endif
  delete (void*)0; // expected-warning {{cannot delete expression with pointer-to-'void' type 'void *'}}
  delete (T*)0; // expected-warning {{deleting pointer to incomplete type}}
  ::S::delete (int*)0; // expected-error {{expected unqualified-id}}
}

struct X0 { };

struct X1 {
  operator int*();
  operator float();
};

struct X2 {
  operator int*(); // expected-note {{conversion}}
  operator float*(); // expected-note {{conversion}}
};

void test_delete_conv(X0 x0, X1 x1, X2 x2) {
  delete x0; // expected-error{{cannot delete}}
  delete x1;
  delete x2; // expected-error{{ambiguous conversion of delete expression of type 'X2' to a pointer}}
}

// PR4782
class X3 {
public:
  static void operator delete(void * mem, size_t size);
};

class X4 {
public:
  static void release(X3 *x);
  static void operator delete(void * mem, size_t size);
};


void X4::release(X3 *x) {
  delete x;
}

class X5 {
public:
  void Destroy() const { delete this; }
};

class Base {
public:
  static void *operator new(signed char) throw(); // expected-error {{'operator new' takes type size_t}}
  static int operator new[] (size_t) throw(); // expected-error {{operator new[]' must return type 'void *'}}
};

class Tier {};
class Comp : public Tier {};

class Thai : public Base {
public:
  Thai(const Tier *adoptDictionary);
};

void loadEngineFor() {
  const Comp *dict;
  new Thai(dict);
}

template <class T> struct TBase {
  void* operator new(T size, int); // expected-error {{'operator new' cannot take a dependent type as first parameter; use size_t}}
};

TBase<int> t1;

class X6 {
public:
  static void operator delete(void*, int); // expected-note {{member found by ambiguous name lookup}}
};

class X7 {
public:
  static void operator delete(void*, int); // expected-note {{member found by ambiguous name lookup}}
};

class X8 : public X6, public X7 {
};

void f(X8 *x8) {
  delete x8; // expected-error {{member 'operator delete' found in multiple base classes of different types}}
}

class X9 {
public:
  static void operator delete(void*, int); // expected-note {{'operator delete' declared here}}
  static void operator delete(void*, float); // expected-note {{'operator delete' declared here}}
};

void f(X9 *x9) {
  delete x9; // expected-error {{no suitable member 'operator delete' in 'X9'}}
}

struct X10 {
  virtual ~X10();
#if __cplusplus >= 201103L
  // expected-note@-2 {{overridden virtual function is here}}
#endif
};

struct X11 : X10 {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'X11'}}
#else
// expected-error@-4 {{deleted function '~X11' cannot override a non-deleted function}}
// expected-note@-5 2 {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif
  void operator delete(void*, int);
#if __cplusplus <= 199711L
  // expected-note@-2 {{'operator delete' declared here}}
#endif
};

void f() {
  X11 x11;
#if __cplusplus <= 199711L
  // expected-note@-2 {{implicit destructor for 'X11' first required here}}
#else
  // expected-error@-4 {{attempt to use a deleted function}}
#endif
}

struct X12 {
  void* operator new(size_t, void*);
};

struct X13 : X12 {
  using X12::operator new;
};

static void* f(void* g)
{
    return new (g) X13();
}

class X14 {
public:
  static void operator delete(void*, const size_t);
};

void f(X14 *x14a, X14 *x14b) {
  delete x14a;
}

class X15 {
private:
  X15(); // expected-note {{declared private here}}
  ~X15(); // expected-note {{declared private here}}
};

void f(X15* x) {
  new X15(); // expected-error {{calling a private constructor}}
  delete x; // expected-error {{calling a private destructor}}
}

namespace PR5918 { // Look for template operator new overloads.
  struct S { template<typename T> static void* operator new(size_t, T); };
  void test() {
    (void)new(0) S;
  }
}

namespace Test1 {

void f() {
  (void)new int[10](1, 2); // expected-error {{array 'new' cannot have initialization arguments}}

  typedef int T[10];
  (void)new T(1, 2); // expected-error {{array 'new' cannot have initialization arguments}}
}

template<typename T>
void g(unsigned i) {
  (void)new T[1](i); // expected-error {{array 'new' cannot have initialization arguments}}
}

template<typename T>
void h(unsigned i) {
  (void)new T(i); // expected-error {{array 'new' cannot have initialization arguments}}
}
template void h<unsigned>(unsigned);
template void h<unsigned[10]>(unsigned); // expected-note {{in instantiation of function template specialization 'Test1::h<unsigned int[10]>' requested here}}

}

// Don't diagnose access for overload candidates that aren't selected.
namespace PR7436 {
struct S1 {
  void* operator new(size_t);
  void operator delete(void* p);

private:
  void* operator new(size_t, void*); // expected-note {{declared private here}}
  void operator delete(void*, void*);
};
class S2 {
  void* operator new(size_t); // expected-note {{declared private here}}
  void operator delete(void* p); // expected-note {{declared private here}}
};

void test(S1* s1, S2* s2) {
  delete s1;
  delete s2; // expected-error {{is a private member}}
  (void)new S1();
  (void)new (0L) S1(); // expected-error {{is a private member}}
  (void)new S2(); // expected-error {{is a private member}}
}
}

namespace rdar8018245 {
  struct X0 {
    static const int value = 17;
  };

  const int X0::value;

  struct X1 {
    static int value;
  };

  int X1::value;

  template<typename T>
  int *f() {
    return new (int[T::value]); // expected-warning{{when type is in parentheses, array cannot have dynamic size}}
  }

  template int *f<X0>();
  template int *f<X1>(); // expected-note{{in instantiation of}}

}

namespace Instantiate {
  template<typename T> struct X {
    operator T*();
  };

  void f(X<int> &xi) {
    delete xi;
  }
}

namespace PR7810 {
  struct X {
    // cv is ignored in arguments
    static void operator delete(void *const);
  };
  struct Y {
    // cv is ignored in arguments
#if __cplusplus < 202002L
    static void operator delete(void *volatile);
#else
    static void operator delete(void *);
#endif
  };
}

// Don't crash on template delete operators
namespace TemplateDestructors {
  struct S {
    virtual ~S() {}

    void* operator new(const size_t size);
    template<class T> void* operator new(const size_t, const int, T*);
    void operator delete(void*, const size_t);
    template<class T> void operator delete(void*, const size_t, const int, T*);
  };
}

namespace DeleteParam {
  struct X {
    void operator delete(X*); // expected-error{{first parameter of 'operator delete' must have type 'void *'}}
  };

  struct Y {
    void operator delete(void* const);
  };
}

// Test that the correct 'operator delete' is selected to pair with
// the unexpected placement 'operator new'.
namespace PairedDelete {
  template <class T> struct A {
    A();
    void *operator new(size_t s, double d = 0);
    void operator delete(void *p, double d);
    void operator delete(void *p) {
      T::dealloc(p);
    }
  };

  A<int> *test() {
    return new A<int>();
  }
}

namespace PR7702 {
  void test1() {
    new DoesNotExist; // expected-error {{unknown type name 'DoesNotExist'}}
  }
}

namespace ArrayNewNeedsDtor {
  struct A { A(); private: ~A(); };
#if __cplusplus <= 199711L
  // expected-note@-2 {{declared private here}}
#endif
  struct B { B(); A a; };
#if __cplusplus <= 199711L
  // expected-error@-2 {{field of type 'A' has private destructor}}
#else
  // expected-note@-4 {{destructor of 'B' is implicitly deleted because field 'a' has an inaccessible destructor}}
#endif

  B *test9() {
    return new B[5];
#if __cplusplus <= 199711L
    // expected-note@-2 {{implicit destructor for 'ArrayNewNeedsDtor::B' first required here}}
#else
    // expected-error@-4 {{attempt to use a deleted function}}
#endif
  }
}

namespace DeleteIncompleteClass {
  struct A; // expected-note {{forward declaration}}
  extern A x;
  void f() { delete x; } // expected-error {{deleting incomplete class type}}
}

namespace DeleteIncompleteClassPointerError {
  struct A; // expected-note {{forward declaration}}
  void f(A *x) { 1+delete x; } // expected-warning {{deleting pointer to incomplete type}} \
                               // expected-error {{invalid operands to binary expression}}
}

namespace PR10504 {
  struct A {
    virtual void foo() = 0;
  };
  void f(A *x) { delete x; } // expected-warning {{delete called on 'PR10504::A' that is abstract but has non-virtual destructor}}
}

struct PlacementArg {};
inline void *operator new[](size_t, const PlacementArg &) throw () {
  return 0;
}
inline void operator delete[](void *, const PlacementArg &) throw () {
}

namespace r150682 {

  template <typename X>
  struct S {
    struct Inner {};
    S() { new Inner[1]; }
  };

  struct T {
  };

  template<typename X>
  void tfn() {
    new (*(PlacementArg*)0) T[1]; // expected-warning 2 {{binding dereferenced null pointer to reference has undefined behavior}}
  }

  void fn() {
    tfn<int>();  // expected-note {{in instantiation of function template specialization 'r150682::tfn<int>' requested here}}
  }

}

namespace P12023 {
  struct CopyCounter
  {
      CopyCounter();
      CopyCounter(const CopyCounter&);
  };

  int main()
  {
    CopyCounter* f = new CopyCounter[10](CopyCounter()); // expected-error {{cannot have initialization arguments}}
      return 0;
  }
}

namespace PR12061 {
  template <class C> struct scoped_array {
    scoped_array(C* p = __null);
  };
  template <class Payload> struct Foo {
    Foo() : a_(new scoped_array<int>[5]) { }
    scoped_array< scoped_array<int> > a_;
  };
  class Bar {};
  Foo<Bar> x;

  template <class C> struct scoped_array2 {
    scoped_array2(C* p = __null, C* q = __null);
  };
  template <class Payload> struct Foo2 {
    Foo2() : a_(new scoped_array2<int>[5]) { }
    scoped_array2< scoped_array2<int> > a_;
  };
  class Bar2 {};
  Foo2<Bar2> x2;

  class MessageLoop {
  public:
    explicit MessageLoop(int type = 0);
  };
  template <class CookieStoreTestTraits>
  class CookieStoreTest {
  protected:
    CookieStoreTest() {
      new MessageLoop;
    }
  };
  struct CookieMonsterTestTraits {
  };
  class DeferredCookieTaskTest : public CookieStoreTest<CookieMonsterTestTraits>
  {
    DeferredCookieTaskTest() {}
  };
}

class DeletingPlaceholder {
  int* f() {
    delete f; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
    return 0;
  }
  int* g(int, int) {
    delete g; // expected-error {{reference to non-static member function must be called}}
    return 0;
  }
};

namespace PR18544 {
  inline void *operator new(size_t); // expected-error {{'operator new' cannot be declared inside a namespace}}
}

// PR19968
inline void* operator new(); // expected-error {{'operator new' must have at least one parameter}}

namespace {
template <class C>
struct A {
  void f() { this->::new; } // expected-error {{expected unqualified-id}}
  void g() { this->::delete; } // expected-error {{expected unqualified-id}}
};
}

#if __cplusplus >= 201103L
template<typename ...T> int *dependent_array_size(T ...v) {
  return new int[]{v...}; // expected-error {{cannot initialize}}
}
int *p0 = dependent_array_size();
int *p3 = dependent_array_size(1, 2, 3);
int *fail = dependent_array_size("hello"); // expected-note {{instantiation of}}
#endif

// FIXME: Our behavior here is incredibly inconsistent. GCC allows
// constant-folding in array bounds in new-expressions.
int (*const_fold)[12] = new int[3][&const_fold + 12 - &const_fold];
#if __cplusplus >= 201402L
// expected-error@-2 {{array size is not a constant expression}}
// expected-note@-3 {{cannot refer to element 12 of non-array}}
#elif __cplusplus < 201103L
// expected-error@-5 {{cannot allocate object of variably modified type}}
#endif
