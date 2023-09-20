// RUN: %clang_cc1 -std=c++17 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -Wc++11-extensions -Wno-long-long -verify -fms-extensions -fexceptions -fcxx-exceptions -DTEST1
// RUN: %clang_cc1 -std=c++98 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -Wc++11-extensions -Wno-long-long -verify=expected,precxx17 -fms-extensions -fexceptions -fcxx-exceptions -DTEST1
// RUN: %clang_cc1 -std=c++11 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -Wc++11-extensions -Wno-long-long -verify=expected,precxx17 -fms-extensions -fexceptions -fcxx-exceptions -DTEST1
// RUN: %clang_cc1 -std=c++14 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -Wc++11-extensions -Wno-long-long -verify=expected,precxx17 -fexceptions -fcxx-exceptions -DTEST2
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -std=c++11 -fms-compatibility -verify -DTEST3

#if TEST1

// MSVC allows type definition in anonymous union and struct
struct A
{
  union
  {
    int a;
    struct B  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    {
      int c;
    } d;

    union C   // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    {
      int e;
      int ee;
    } f;

    typedef int D;  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    struct F;  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
  };

  struct
  {
    int a2;

    struct B2  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    {
      int c2;
    } d2;

	union C2  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    {
      int e2;
      int ee2;
    } f2;

    typedef int D2;  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    struct F2;  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
  };
};

// __stdcall handling
struct M {
    int __stdcall addP();
    float __stdcall subtractP();
};

// __unaligned handling
typedef char __unaligned *aligned_type;
typedef struct UnalignedTag { int f; } __unaligned *aligned_type2;
typedef char __unaligned aligned_type3;

struct aligned_type4 {
  int i;
};

__unaligned int aligned_type4::*p1_aligned_type4 = &aligned_type4::i;
int aligned_type4::* __unaligned p2_aligned_type4 = &aligned_type4::i;
__unaligned int aligned_type4::* __unaligned p3_aligned_type4 = &aligned_type4::i;
void (aligned_type4::*__unaligned p4_aligned_type4)();

// Check that __unaligned qualifier can be used for overloading
void foo_unaligned(int *arg) {} // expected-note {{not viable: 1st argument ('const __unaligned int *') would lose const qualifier}}
void foo_unaligned(__unaligned int *arg) {} // expected-note {{not viable: 1st argument ('const __unaligned int *') would lose const qualifier}}
void foo_unaligned(int arg) {} // expected-note {{previous definition is here}}
void foo_unaligned(__unaligned int arg) {} // expected-error {{redefinition of 'foo_unaligned'}} \
                                           // expected-note {{not viable: no known conversion from 'const __unaligned int *' to '__unaligned int'}}
class A_unaligned {};
class B_unaligned : public A_unaligned {};
int foo_unaligned(__unaligned A_unaligned *arg) { return 0; } // expected-note {{not viable: no known conversion from 'const __unaligned int *' to '__unaligned A_unaligned *'}}
void *foo_unaligned(B_unaligned *arg) { return 0; } // expected-note {{not viable: no known conversion from 'const __unaligned int *' to 'B_unaligned *'}}

void test_unaligned() {
  int *p1 = 0;
  foo_unaligned(p1);

  const __unaligned int *const_p1 = 0;
  foo_unaligned(const_p1); // expected-error {{no matching function for call to 'foo_unaligned'}}

  __unaligned int *p2 = 0;
  foo_unaligned(p2);

  __unaligned B_unaligned *p3 = 0;
  int p4 = foo_unaligned(p3); // expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'void *'}}
  // expected-warning@-1 {{implicit cast from type '__unaligned B_unaligned *' to type 'B_unaligned *' drops __unaligned qualifier}}

  B_unaligned *p5 = p3;
  // expected-warning@-1 {{implicit cast from type '__unaligned B_unaligned *' to type 'B_unaligned *' drops __unaligned qualifier}}

  __unaligned B_unaligned *p6 = p3;

  p1_aligned_type4 = p2_aligned_type4;
  p2_aligned_type4 = p1_aligned_type4;
  // expected-warning@-1 {{implicit cast from type '__unaligned int aligned_type4::*' to type 'int aligned_type4::*' drops __unaligned qualifier}}
  p3_aligned_type4 = p1_aligned_type4;

  __unaligned int a[10];
  int *b = a;
  // expected-warning@-1 {{implicit cast from type '__unaligned int[10]' to type 'int *' drops __unaligned qualifier}}
}

// Test from PR27367
// We should accept assignment of an __unaligned pointer to a non-__unaligned
// pointer to void
typedef struct _ITEMIDLIST { int i; } ITEMIDLIST;
typedef ITEMIDLIST __unaligned *LPITEMIDLIST;
extern "C" __declspec(dllimport) void __stdcall CoTaskMemFree(void* pv);
__inline void FreeIDListArray(LPITEMIDLIST *ppidls) {
  CoTaskMemFree(*ppidls);
  __unaligned int *x = 0;
  void *y = x;
}

// Test from PR27666
// We should accept type conversion of __unaligned to non-__unaligned references
typedef struct in_addr {
public:
  in_addr(in_addr &a) {} // precxx17-note {{candidate constructor not viable: expects an lvalue for 1st argument}}
  in_addr(in_addr *a) {} // precxx17-note {{candidate constructor not viable: no known conversion from 'IN_ADDR' (aka 'in_addr') to 'in_addr *' for 1st argument}}
} IN_ADDR;

void f(IN_ADDR __unaligned *a) {
  IN_ADDR local_addr = *a;
  // FIXME: MSVC accepts the following; not sure why clang tries to
  // copy-construct an in_addr.
  IN_ADDR local_addr2 = a; // precxx17-error {{no viable constructor copying variable of type 'IN_ADDR' (aka 'in_addr')}}
  // expected-warning@-1 {{implicit cast from type '__unaligned IN_ADDR *' (aka '__unaligned in_addr *') to type 'in_addr *' drops __unaligned qualifier}}
  IN_ADDR local_addr3(a);
  // expected-warning@-1 {{implicit cast from type '__unaligned IN_ADDR *' (aka '__unaligned in_addr *') to type 'in_addr *' drops __unaligned qualifier}}
}

template<typename T> void h1(T (__stdcall M::* const )()) { }

void m1() {
  h1<int>(&M::addP);
  h1(&M::subtractP);
}


namespace signed_hex_i64 {
void f(long long); // expected-note {{candidate function}}
void f(int); // expected-note {{candidate function}}
void g() {
  // This used to be controlled by -fms-extensions, but it is now under
  // -fms-compatibility.
  f(0xffffffffffffffffLL); // expected-error {{call to 'f' is ambiguous}}
  f(0xffffffffffffffffi64);
}
}

// Enumeration types with a fixed underlying type.
const int seventeen = 17;
typedef int Int;

struct X0 {
  enum E1 : Int { SomeOtherValue } field;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{enumeration types with a fixed underlying type are a C++11 extension}}
#endif

  enum E1 : seventeen;
#if __cplusplus >= 201103L
  // expected-error@-2 {{bit-field}}
#endif
};

#if __cplusplus <= 199711L
// expected-warning@+2 {{enumeration types with a fixed underlying type are a C++11 extension}}
#endif
enum : long long {
  SomeValue = 0x100000000
};


class AAA {
__declspec(dllimport) void f(void) { }
void f2(void); // expected-note{{previous declaration is here}}
};

__declspec(dllimport) void AAA::f2(void) { // expected-error{{dllimport cannot be applied to non-inline function definition}}
                                           // expected-error@-1{{redeclaration of 'AAA::f2' cannot add 'dllimport' attribute}}

}



template <class T>
class BB {
public:
   void f(int g = 10 ); // expected-note {{previous definition is here}}
};

template <class T>
void BB<T>::f(int g = 0) { } // expected-warning {{redefinition of default argument}}



extern void static_func();
void static_func(); // expected-note {{previous declaration is here}}


static void static_func() // expected-warning {{redeclaring non-static 'static_func' as static is a Microsoft extension}}
{

}

extern const int static_var; // expected-note {{previous declaration is here}}
static const int static_var = 3; // expected-warning {{redeclaring non-static 'static_var' as static is a Microsoft extension}}

void pointer_to_integral_type_conv(char* ptr) {
  char ch = (char)ptr;   // expected-warning {{cast to smaller integer type 'char' from 'char *'}}
  short sh = (short)ptr; // expected-warning {{cast to smaller integer type 'short' from 'char *'}}
  ch = (char)ptr;        // expected-warning {{cast to smaller integer type 'char' from 'char *'}}
  sh = (short)ptr;       // expected-warning {{cast to smaller integer type 'short' from 'char *'}}

  // These are valid C++.
  bool b = (bool)ptr;
  b = static_cast<bool>(ptr);

  // This is bad.
  b = reinterpret_cast<bool>(ptr); // expected-error {{cast from pointer to smaller type 'bool' loses information}}
}

void void_pointer_to_integral_type_conv(void *ptr) {
  char ch = (char)ptr;   // expected-warning {{cast to smaller integer type 'char' from 'void *'}}
  short sh = (short)ptr; // expected-warning {{cast to smaller integer type 'short' from 'void *'}}
  ch = (char)ptr;        // expected-warning {{cast to smaller integer type 'char' from 'void *'}}
  sh = (short)ptr;       // expected-warning {{cast to smaller integer type 'short' from 'void *'}}

  // These are valid C++.
  bool b = (bool)ptr;
  b = static_cast<bool>(ptr);

  // This is bad.
  b = reinterpret_cast<bool>(ptr); // expected-error {{cast from pointer to smaller type 'bool' loses information}}
}

struct PR11150 {
  class X {
    virtual void f() = 0;
  };

  int array[__is_abstract(X)? 1 : -1];
};

void f() { int __except = 0; }

void ::f(); // expected-warning{{extra qualification on member 'f'}}

class C {
  C::C(); // expected-warning{{extra qualification on member 'C'}}
};

struct StructWithProperty {
  __declspec(property(get=GetV)) int V1;
  __declspec(property(put=SetV)) int V2;
  __declspec(property(get=GetV, put=SetV_NotExist)) int V3;
  __declspec(property(get=GetV_NotExist, put=SetV)) int V4;
  __declspec(property(get=GetV, put=SetV)) int V5;

  int GetV() { return 123; }
  void SetV(int i) {}
};
void TestProperty() {
  StructWithProperty sp;
  int i = sp.V2; // expected-error{{no getter defined for property 'V2'}}
  sp.V1 = 12; // expected-error{{no setter defined for property 'V1'}}
  int j = sp.V4; // expected-error{{no member named 'GetV_NotExist' in 'StructWithProperty'}} expected-error{{cannot find suitable getter for property 'V4'}}
  sp.V3 = 14; // expected-error{{no member named 'SetV_NotExist' in 'StructWithProperty'}} expected-error{{cannot find suitable setter for property 'V3'}}
  int k = sp.V5;
  sp.V5 = k++;
}

/* 4 tests for PseudoObject, begin */
struct SP1
{
  bool operator()() { return true; }
};
struct SP2
{
  __declspec(property(get=GetV)) SP1 V;
  SP1 GetV() { return SP1(); }
};
void TestSP2() {
  SP2 sp2;
  bool b = sp2.V();
}

struct SP3 {
  template <class T>
  void f(T t) {}
};
template <class T>
struct SP4
{
  __declspec(property(get=GetV)) int V;
  int GetV() { return 123; }
  void f() { SP3 s2; s2.f(V); }
};
void TestSP4() {
  SP4<int> s;
  s.f();
}

template <class T>
struct SP5
{
  __declspec(property(get=GetV)) T V;
  int GetV() { return 123; }
  void f() { int *p = new int[V]; }
};

template <class T>
struct SP6
{
public:
  __declspec(property(get=GetV)) T V;
  T GetV() { return 123; }
  void f() { int t = V; }
};
void TestSP6() {
  SP6<int> c;
  c.f();
}
/* 4 tests for PseudoObject, end */

// Property access: explicit, implicit, with Qualifier
struct SP7 {
  __declspec(property(get=GetV, put=SetV)) int V;
  int GetV() { return 123; }
  void SetV(int v) {}

  void ImplicitAccess() { int i = V; V = i; }
  void ExplicitAccess() { int i = this->V; this->V = i; }
};
struct SP8: public SP7 {
  void AccessWithQualifier() { int i = SP7::V; SP7::V = i; }
};

// Property usage
template <class T>
struct SP9 {
  __declspec(property(get=GetV, put=SetV)) T V;
  T GetV() { return 0; }
  void SetV(T v) {}
  bool f() { V = this->V; return V < this->V; }
  void g() { V++; }
  void h() { V*=2; }
};
struct SP10 {
  SP10(int v) {}
  bool operator<(const SP10& v) { return true; }
  SP10 operator*(int v) { return *this; }
  SP10 operator+(int v) { return *this; }
  SP10& operator=(const SP10& v) { return *this; }
};
void TestSP9() {
  SP9<int> c;
  int i = c.V; // Decl initializer
  i = c.V; // Binary op operand
  c.SetV(c.V); // CallExpr arg
  int *p = new int[c.V + 1]; // Array size
  p[c.V] = 1; // Array index

  c.V = 123; // Setter

  c.V++; // Unary op operand
  c.V *= 2; // Unary op operand

  SP9<int*> c2;
  c2.V[0] = 123; // Array

  SP9<SP10> c3;
  c3.f(); // Overloaded binary op operand
  c3.g(); // Overloaded incdec op operand
  c3.h(); // Overloaded unary op operand
}

union u {
  int *i1;
  int &i2;  // expected-warning {{union member 'i2' has reference type 'int &', which is a Microsoft extension}}
};

// Property getter using reference.
struct SP11 {
  __declspec(property(get=GetV)) int V;
  int _v;
  int& GetV() { return _v; }
  void UseV();
  void TakePtr(int *) {}
  void TakeRef(int &) {}
  void TakeVal(int) {}
};

void SP11::UseV() {
  TakePtr(&V);
  TakeRef(V);
  TakeVal(V);
}

struct StructWithUnnamedMember {
  __declspec(property(get=GetV)) int : 10; // expected-error {{anonymous property is not supported}}
};

struct MSPropertyClass {
  int get() { return 42; }
  int __declspec(property(get = get)) n;
};

int *f(MSPropertyClass &x) {
  return &x.n; // expected-error {{address of property expression requested}}
}
int MSPropertyClass::*g() {
  return &MSPropertyClass::n; // expected-error {{address of property expression requested}}
}

namespace rdar14250378 {
  class Bar {};

  namespace NyNamespace {
    class Foo {
    public:
      Bar* EnsureBar();
    };

    class Baz : public Foo {
    public:
      friend class Bar;
    };

    Bar* Foo::EnsureBar() {
      return 0;
    }
  }
}

// expected-error@+1 {{'sealed' keyword not permitted with interface types}}
__interface InterfaceWithSealed sealed {
};

struct SomeBase {
  virtual void OverrideMe();

  // expected-note@+2 {{overridden virtual function is here}}
  // expected-warning@+1 {{'sealed' keyword is a Microsoft extension}}
  virtual void SealedFunction() sealed; // expected-note {{overridden virtual function is here}}
};

// expected-note@+2 {{'SealedType' declared here}}
// expected-warning@+1 {{'sealed' keyword is a Microsoft extension}}
struct SealedType sealed : SomeBase {
  // expected-error@+2 {{declaration of 'SealedFunction' overrides a 'sealed' function}}
  // FIXME. warning can be suppressed if we're also issuing error for overriding a 'final' function.
  virtual void SealedFunction(); // expected-warning {{'SealedFunction' overrides a member function but is not marked 'override'}}

#if __cplusplus <= 199711L
  // expected-warning@+2 {{'override' keyword is a C++11 extension}}
#endif
  virtual void OverrideMe() override;
};

// expected-error@+1 {{base 'SealedType' is marked 'sealed'}}
struct InheritFromSealed : SealedType {};

class SealedDestructor { // expected-note {{mark 'SealedDestructor' as 'sealed' to silence this warning}}
    // expected-warning@+1 {{'sealed' keyword is a Microsoft extension}}
    virtual ~SealedDestructor() sealed; // expected-warning {{class with destructor marked 'sealed' cannot be inherited from}}
};

// expected-warning@+1 {{'abstract' keyword is a Microsoft extension}}
class AbstractClass abstract {
  int i;
};

// expected-error@+1 {{variable type 'AbstractClass' is an abstract class}}
AbstractClass abstractInstance;

// expected-warning@+4 {{abstract class is marked 'sealed'}}
// expected-note@+3 {{'AbstractAndSealedClass' declared here}}
// expected-warning@+2 {{'abstract' keyword is a Microsoft extension}}
// expected-warning@+1 {{'sealed' keyword is a Microsoft extension}}
class AbstractAndSealedClass abstract sealed {}; // Does no really make sense, but allowed

// expected-error@+1 {{variable type 'AbstractAndSealedClass' is an abstract class}}
AbstractAndSealedClass abstractAndSealedInstance;
// expected-error@+1 {{base 'AbstractAndSealedClass' is marked 'sealed'}}
class InheritFromAbstractAndSealed : AbstractAndSealedClass {};

#if __cplusplus <= 199711L
// expected-warning@+4 {{'final' keyword is a C++11 extension}}
// expected-warning@+3 {{'final' keyword is a C++11 extension}}
#endif
// expected-error@+1 {{class already marked 'final'}}
class TooManyVirtSpecifiers1 final final {};
#if __cplusplus <= 199711L
// expected-warning@+4 {{'final' keyword is a C++11 extension}}
#endif
// expected-warning@+2 {{'sealed' keyword is a Microsoft extension}}
// expected-error@+1 {{class already marked 'sealed'}}
class TooManyVirtSpecifiers2 final sealed {};
#if __cplusplus <= 199711L
// expected-warning@+6 {{'final' keyword is a C++11 extension}}
// expected-warning@+5 {{'final' keyword is a C++11 extension}}
#endif
// expected-warning@+3 {{abstract class is marked 'final'}}
// expected-warning@+2 {{'abstract' keyword is a Microsoft extension}}
// expected-error@+1 {{class already marked 'final'}}
class TooManyVirtSpecifiers3 final abstract final {};
#if __cplusplus <= 199711L
// expected-warning@+6 {{'final' keyword is a C++11 extension}}
#endif
// expected-warning@+4 {{abstract class is marked 'final'}}
// expected-warning@+3 {{'abstract' keyword is a Microsoft extension}}
// expected-warning@+2 {{'abstract' keyword is a Microsoft extension}}
// expected-error@+1 {{class already marked 'abstract'}}
class TooManyVirtSpecifiers4 abstract final abstract {};

class Base {
  virtual void i();
};
class AbstractFunctionInClass : public Base {
  // expected-note@+2 {{unimplemented pure virtual method 'f' in 'AbstractFunctionInClass'}}
  // expected-warning@+1 {{'abstract' keyword is a Microsoft extension}}
  virtual void f() abstract;
  // expected-warning@+1 {{'abstract' keyword is a Microsoft extension}}
  void g() abstract; // expected-error {{'g' is not virtual and cannot be declared pure}}
  // expected-note@+2 {{unimplemented pure virtual method 'h' in 'AbstractFunctionInClass'}}
  // expected-warning@+1 {{'abstract' keyword is a Microsoft extension}}
  virtual void h() abstract = 0; // expected-error {{class member already marked 'abstract'}}
#if __cplusplus <= 199711L
  // expected-warning@+4 {{'override' keyword is a C++11 extension}}
#endif
  // expected-note@+2 {{unimplemented pure virtual method 'i' in 'AbstractFunctionInClass'}}
  // expected-warning@+1 {{'abstract' keyword is a Microsoft extension}}
  virtual void i() abstract override;
};

// expected-error@+1 {{variable type 'AbstractFunctionInClass' is an abstract class}}
AbstractFunctionInClass abstractFunctionInClassInstance;

void AfterClassBody() {
  // expected-warning@+1 {{attribute 'deprecated' is ignored, place it after "struct" to apply attribute to type declaration}}
  struct D {} __declspec(deprecated);

  struct __declspec(align(4)) S {} __declspec(align(8)) s1;
  S s2;
  _Static_assert(__alignof(S) == 4, "");
  _Static_assert(__alignof(s1) == 8, "");
  _Static_assert(__alignof(s2) == 4, "");
}

namespace PR24246 {
template <typename TX> struct A {
  template <bool> struct largest_type_select;
  template <> struct largest_type_select<false> {
    blah x;  // expected-error {{unknown type name 'blah'}}
  };
};
}

class PR34109_class {
  PR34109_class() {}
  virtual ~PR34109_class() {}
};

void operator delete(void *) throw();
// expected-note@-1 {{previous declaration is here}}
__declspec(dllexport) void operator delete(void *) throw();
// expected-error@-1  {{redeclaration of 'operator delete' cannot add 'dllexport' attribute}}

void PR34109(int* a) {
  delete a;
}

namespace PR42089 {
  struct S {
    __attribute__((nothrow)) void Foo(); // expected-note {{previous declaration is here}}
    __attribute__((nothrow)) void Bar();
  };
  void S::Foo(){} // expected-warning {{is missing exception specification}}
  __attribute__((nothrow)) void S::Bar(){}
}

namespace UnalignedConv {
struct S {
  bool operator==(int) const;
};

int func() {
  S __unaligned s;
  return s == 42;
}
}


#elif TEST2

// Check that __unaligned is not recognized if MS extensions are not enabled
typedef char __unaligned *aligned_type; // expected-error {{expected ';' after top level declarator}}

#elif TEST3

namespace PR32750 {
template<typename T> struct A {};
template<typename T> struct B : A<A<T>> { A<T>::C::D d; }; // expected-warning {{implicit 'typename' is a C++20 extension}}
}

#else

#error Unknown test mode

#endif

