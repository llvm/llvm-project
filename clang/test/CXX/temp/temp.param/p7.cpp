// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 -std=c++98 %s -Wno-c++11-extensions
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

// C++98:
// A non-type template-parameter shall not be declared to have
// floating point, class, or void type.
struct A; // expected-note {{forward declaration}}

template<double d> class X; // cxx17-error{{cannot have type}}
template<double* pd> class Y; //OK 
template<double& rd> class Z; //OK 

template<A a> class X0; // expected-error{{has incomplete type 'A'}}

struct A {};

template<A a> class X0b; // cxx17-error{{cannot have type 'A' before C++20}}

typedef void VOID;
template<VOID a> class X01; // expected-error{{has incomplete type 'VOID'}}

// C++11 disallows rvalue references.

template<int &R> struct lval_ref;
template<int &&R> struct rval_ref; // expected-warning 0-1{{extension}} expected-error {{non-type template parameter has rvalue reference type 'int &&'}}

// C++20 requires a structural type. In addition to the above cases, this allows:

// arbitrary scalar types; we generally include complex types in that list
template<_Complex float ci> struct ComplexFloat; // cxx17-error {{cannot have type '_Complex float' before C++20}}
template<_Complex int ci> struct ComplexInt; // cxx17-error {{cannot have type '_Complex int' before C++20}}
template<_ExtInt(42) ei> struct ExtInt;

// atomic types aren't scalar types
template<_Atomic float ci> struct AtomicFloat; // expected-error {{cannot have type '_Atomic(float)'}}
template<_Atomic int ci> struct AtomicInt; // expected-error {{cannot have type '_Atomic(int)'}}

// we allow vector types as an extension
typedef __attribute__((ext_vector_type(4))) int VI4;
typedef __attribute__((ext_vector_type(4))) float VF4;
template<VI4> struct VectorInt; // cxx17-error {{cannot have type 'VI4'}}
template<VF4> struct VectorFloat; // cxx17-error {{cannot have type 'VF4'}}

struct A2 {};

struct RRef {
  int &&r; // cxx20-note 1+{{'RRef' is not a structural type because it has a non-static data member of rvalue reference type}}
};

// class types with all public members and bases, no mutable state, and no rvalue references.
struct B : A, public A2 {
  int a;
private:
  void f();
  static int s;
public:
  float g;
  int &r = a;
  void *p;
  A2 a2;
  RRef *ptr_to_bad;
  RRef &ref_to_bad = *ptr_to_bad;
  _Complex int ci;
  _Complex float cf;
  _ExtInt(42) ei;
  VI4 vi4;
  VF4 vf4;
};

template<B> struct ClassNTTP {}; // cxx17-error {{cannot have type 'B'}}

template<RRef> struct WithRRef {}; // cxx17-error {{cannot have type 'RRef'}}
// cxx20-error@-1 {{type 'RRef' of non-type template parameter is not a structural type}}

struct BadBase
  : RRef {}; // cxx20-note {{'BadBase' is not a structural type because it has a base class of non-structural type 'RRef'}}
template<BadBase> struct WithBadBase {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct BadField {
  RRef r; // cxx20-note {{'BadField' is not a structural type because it has a non-static data member of non-structural type 'RRef'}}
};
template<BadField> struct WithBadField {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct BadFieldArray {
  RRef r[3][2]; // cxx20-note {{'BadFieldArray' is not a structural type because it has a non-static data member of non-structural type 'RRef'}}
};
template<BadFieldArray> struct WithBadFieldArray {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct ProtectedBase
  : protected A {}; // cxx20-note {{'ProtectedBase' is not a structural type because it has a base class that is not public}}
template<ProtectedBase> struct WithProtectedBase {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct PrivateBase
  : private A {}; // cxx20-note {{'PrivateBase' is not a structural type because it has a base class that is not public}}
template<PrivateBase> struct WithPrivateBase {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

class Private2Base
  : A {}; // cxx20-note {{'Private2Base' is not a structural type because it has a base class that is not public}}
template<Private2Base> struct WithPrivate2Base {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct ProtectedField {
protected:
  A r; // cxx20-note {{'ProtectedField' is not a structural type because it has a non-static data member that is not public}}
};
template<ProtectedField> struct WithProtectedField {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct PrivateField {
private:
  A r; // cxx20-note {{'PrivateField' is not a structural type because it has a non-static data member that is not public}}
};
template<PrivateField> struct WithPrivateField {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

class Private2Field {
  A r; // cxx20-note {{'Private2Field' is not a structural type because it has a non-static data member that is not public}}
};
template<Private2Field> struct WithPrivate2Field {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

struct MutableField {
  mutable int n; // cxx20-note {{'MutableField' is not a structural type because it has a mutable non-static data member}}
};
template<MutableField> struct WithMutableField {}; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}

template<typename T> struct BadExtType { T t; }; // cxx20-note 2{{has a non-static data member of non-structural type}}
template<BadExtType<_Atomic float> > struct AtomicFloatField; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}
template<BadExtType<_Atomic int> > struct AtomicInt; // cxx17-error {{cannot have type}} cxx20-error {{is not a structural type}}
