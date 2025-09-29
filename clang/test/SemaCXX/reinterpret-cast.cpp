// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding -Wundefined-reinterpret-cast -Wno-unused-volatile-lvalue %s

#include <stdint.h>

enum test { testval = 1 };
struct structure { int m; };
typedef void (*fnptr)();

// Test the conversion to self.
void self_conversion()
{
  // T->T is allowed per [expr.reinterpret.cast]p2 so long as it doesn't
  // cast away constness, and is integral, enumeration, pointer or 
  // pointer-to-member.
  int i = 0;
  (void)reinterpret_cast<int>(i);

  test e = testval;
  (void)reinterpret_cast<test>(e);

  // T*->T* is allowed
  int *pi = 0;
  (void)reinterpret_cast<int*>(pi);

  const int structure::*psi = 0;
  (void)reinterpret_cast<const int structure::*>(psi);

  const int ci = 0;
  (void)reinterpret_cast<const int>(i);

  structure s;
  (void)reinterpret_cast<structure>(s); // expected-error {{reinterpret_cast from 'structure' to 'structure' is not allowed}}

  float f = 0.0f;
  (void)reinterpret_cast<float>(f); // expected-error {{reinterpret_cast from 'float' to 'float' is not allowed}}
}

// Test conversion between pointer and integral types, as in /3 and /4.
void integral_conversion()
{
  void *vp = reinterpret_cast<void*>(testval);
  intptr_t i = reinterpret_cast<intptr_t>(vp);
  (void)reinterpret_cast<float*>(i);
  fnptr fnp = reinterpret_cast<fnptr>(i);
  (void)reinterpret_cast<char>(fnp); // expected-error {{cast from pointer to smaller type 'char' loses information}}
  (void)reinterpret_cast<intptr_t>(fnp);
}

void pointer_conversion()
{
  int *p1 = 0;
  float *p2 = reinterpret_cast<float*>(p1);
  structure *p3 = reinterpret_cast<structure*>(p2);
  typedef int **ppint;
  ppint *deep = reinterpret_cast<ppint*>(p3);
  (void)reinterpret_cast<fnptr*>(deep);
}

void constness()
{
  int ***const ipppc = 0;
  // Valid: T1* -> T2 const*
  int const *icp = reinterpret_cast<int const*>(ipppc);
  // Invalid: T1 const* -> T2*
  (void)reinterpret_cast<int*>(icp); // expected-error {{reinterpret_cast from 'const int *' to 'int *' casts away qualifiers}}
  // Invalid: T1*** -> T2 const* const**
  int const *const **icpcpp = reinterpret_cast<int const* const**>(ipppc); // expected-error {{reinterpret_cast from 'int ***' to 'const int *const **' casts away qualifiers}}
  // Valid: T1* -> T2*
  int *ip = reinterpret_cast<int*>(icpcpp);
  // Valid: T* -> T const*
  (void)reinterpret_cast<int const*>(ip);
  // Valid: T*** -> T2 const* const* const*
  (void)reinterpret_cast<int const* const* const*>(ipppc);

  // C++ [expr.type]/8.2.2:
  //   If a pr-value initially has the type cv-T, where T is a
  //   cv-unqualified non-class, non-array type, the type of the
  //   expression is adjusted to T prior to any further analysis.
  int i = 0;
  // Valid: T -> T (top level const is ignored)
  (void)reinterpret_cast<const int>(i);
  // Valid: T* -> T* (top level const is ignored)
  (void)reinterpret_cast<int *const>(ip);
}

void fnptrs()
{
  typedef int (*fnptr2)(int);
  fnptr fp = 0;
  (void)reinterpret_cast<fnptr2>(fp);
  void *vp = reinterpret_cast<void*>(fp);
  (void)reinterpret_cast<fnptr>(vp);
}

void refs()
{
  long l = 0;
  char &c = reinterpret_cast<char&>(l);
  // Bad: from rvalue
  (void)reinterpret_cast<int&>(&c); // expected-error {{reinterpret_cast from rvalue to reference type 'int &'}}
}

void memptrs()
{
  const int structure::*psi = 0;
  (void)reinterpret_cast<const float structure::*>(psi);
  (void)reinterpret_cast<int structure::*>(psi); // expected-error {{reinterpret_cast from 'const int structure::*' to 'int structure::*' casts away qualifiers}}

  void (structure::*psf)() = 0;
  (void)reinterpret_cast<int (structure::*)()>(psf);

  (void)reinterpret_cast<void (structure::*)()>(psi); // expected-error-re {{reinterpret_cast from 'const int structure::*' to 'void (structure::*)(){{( __attribute__\(\(thiscall\)\))?}}' is not allowed}}
  (void)reinterpret_cast<int structure::*>(psf); // expected-error-re {{reinterpret_cast from 'void (structure::*)(){{( __attribute__\(\(thiscall\)\))?}}' to 'int structure::*' is not allowed}}

  // Cannot cast from integers to member pointers, not even the null pointer
  // literal.
  (void)reinterpret_cast<void (structure::*)()>(0); // expected-error-re {{reinterpret_cast from 'int' to 'void (structure::*)(){{( __attribute__\(\(thiscall\)\))?}}' is not allowed}}
  (void)reinterpret_cast<int structure::*>(0); // expected-error {{reinterpret_cast from 'int' to 'int structure::*' is not allowed}}
}

namespace PR5545 {
// PR5545
class A;
class B;
void (A::*a)();
void (B::*b)() = reinterpret_cast<void (B::*)()>(a);
}

void const_arrays() {
  typedef char STRING[10];
  const STRING *s;
  const char *c;

  (void)reinterpret_cast<char *>(s); // expected-error {{reinterpret_cast from 'const STRING *' (aka 'const char (*)[10]') to 'char *' casts away qualifiers}}
  (void)reinterpret_cast<const STRING *>(c);
}

namespace PR9564 {
  struct a { int a : 10; }; a x;
  int *y = &reinterpret_cast<int&>(x.a); // expected-error {{reinterpret_cast from bit-field lvalue to reference type 'int &'}}

  __attribute((ext_vector_type(4))) typedef float v4;
  float& w(v4 &a) { return reinterpret_cast<float&>(a[1]); } // expected-error {{not allowed}}
}

void dereference_reinterpret_cast() {
  struct A {};
  typedef A A2;
  class B {};
  typedef B B2;
  A a;
  B b;
  A2 a2;
  B2 b2;
  long l;
  double d;
  float f;
  char c;
  unsigned char uc;
  void* v_ptr;
  (void)reinterpret_cast<double&>(l);  // expected-warning {{reinterpret_cast from 'long' to 'double &' has undefined behavior}}
  (void)*reinterpret_cast<double*>(&l);  // expected-warning {{dereference of type 'double *' that was reinterpret_cast from type 'long *' has undefined behavior}}
  (void)reinterpret_cast<double&>(f);  // expected-warning {{reinterpret_cast from 'float' to 'double &' has undefined behavior}}
  (void)*reinterpret_cast<double*>(&f);  // expected-warning {{dereference of type 'double *' that was reinterpret_cast from type 'float *' has undefined behavior}}
  (void)reinterpret_cast<float&>(l);  // expected-warning {{reinterpret_cast from 'long' to 'float &' has undefined behavior}}
  (void)*reinterpret_cast<float*>(&l);  // expected-warning {{dereference of type 'float *' that was reinterpret_cast from type 'long *' has undefined behavior}}
  (void)reinterpret_cast<float&>(d);  // expected-warning {{reinterpret_cast from 'double' to 'float &' has undefined behavior}}
  (void)*reinterpret_cast<float*>(&d);  // expected-warning {{dereference of type 'float *' that was reinterpret_cast from type 'double *' has undefined behavior}}

  // Look through parens
  (void)*(reinterpret_cast<double*>(&l));  // expected-warning {{dereference of type 'double *' that was reinterpret_cast from type 'long *' has undefined behavior}}
  (void)*((reinterpret_cast<double*>((&l))));  // expected-warning {{dereference of type 'double *' that was reinterpret_cast from type 'long *' has undefined behavior}}

  // TODO: add warning for tag types
  (void)reinterpret_cast<A&>(b);
  (void)*reinterpret_cast<A*>(&b);
  (void)reinterpret_cast<B&>(a);
  (void)*reinterpret_cast<B*>(&a);
  (void)reinterpret_cast<A2&>(b2);
  (void)*reinterpret_cast<A2*>(&b2);
  (void)reinterpret_cast<B2&>(a2);
  (void)*reinterpret_cast<B2*>(&a2);

  // Casting to itself is allowed
  (void)reinterpret_cast<A&>(a);
  (void)*reinterpret_cast<A*>(&a);
  (void)reinterpret_cast<B&>(b);
  (void)*reinterpret_cast<B*>(&b);
  (void)reinterpret_cast<long&>(l);
  (void)*reinterpret_cast<long*>(&l);
  (void)reinterpret_cast<double&>(d);
  (void)*reinterpret_cast<double*>(&d);
  (void)reinterpret_cast<char&>(c);
  (void)*reinterpret_cast<char*>(&c);

  // Casting to and from chars are allowable
  (void)reinterpret_cast<A&>(c);
  (void)*reinterpret_cast<A*>(&c);
  (void)reinterpret_cast<B&>(c);
  (void)*reinterpret_cast<B*>(&c);
  (void)reinterpret_cast<long&>(c);
  (void)*reinterpret_cast<long*>(&c);
  (void)reinterpret_cast<double&>(c);
  (void)*reinterpret_cast<double*>(&c);
  (void)reinterpret_cast<char&>(l);
  (void)*reinterpret_cast<char*>(&l);
  (void)reinterpret_cast<char&>(d);
  (void)*reinterpret_cast<char*>(&d);
  (void)reinterpret_cast<char&>(f);
  (void)*reinterpret_cast<char*>(&f);

  // Casting from void pointer.
  (void)*reinterpret_cast<A*>(v_ptr);
  (void)*reinterpret_cast<B*>(v_ptr);
  (void)*reinterpret_cast<long*>(v_ptr);
  (void)*reinterpret_cast<double*>(v_ptr);
  (void)*reinterpret_cast<float*>(v_ptr);

  // Casting to void pointer
  (void)*reinterpret_cast<void*>(&a); // expected-error {{indirection not permitted on operand of type 'void *'}}
  (void)*reinterpret_cast<void*>(&b); // expected-error {{indirection not permitted on operand of type 'void *'}}
  (void)*reinterpret_cast<void*>(&l); // expected-error {{indirection not permitted on operand of type 'void *'}}
  (void)*reinterpret_cast<void*>(&d); // expected-error {{indirection not permitted on operand of type 'void *'}}
  (void)*reinterpret_cast<void*>(&f); // expected-error {{indirection not permitted on operand of type 'void *'}}
}

void reinterpret_cast_allowlist () {
  // the dynamic type of the object
  int a;
  float b;
  (void)reinterpret_cast<int&>(a);
  (void)*reinterpret_cast<int*>(&a);
  (void)reinterpret_cast<float&>(b);
  (void)*reinterpret_cast<float*>(&b);

  // a cv-qualified version of the dynamic object
  (void)reinterpret_cast<const int&>(a);
  (void)*reinterpret_cast<const int*>(&a);
  (void)reinterpret_cast<volatile int&>(a);
  (void)*reinterpret_cast<volatile int*>(&a);
  (void)reinterpret_cast<const volatile int&>(a);
  (void)*reinterpret_cast<const volatile int*>(&a);
  (void)reinterpret_cast<const float&>(b);
  (void)*reinterpret_cast<const float*>(&b);
  (void)reinterpret_cast<volatile float&>(b);
  (void)*reinterpret_cast<volatile float*>(&b);
  (void)reinterpret_cast<const volatile float&>(b);
  (void)*reinterpret_cast<const volatile float*>(&b);

  // a type that is the signed or unsigned type corresponding to the dynamic
  // type of the object
  signed d;
  unsigned e;
  (void)reinterpret_cast<signed&>(d);
  (void)*reinterpret_cast<signed*>(&d);
  (void)reinterpret_cast<signed&>(e);
  (void)*reinterpret_cast<signed*>(&e);
  (void)reinterpret_cast<unsigned&>(d);
  (void)*reinterpret_cast<unsigned*>(&d);
  (void)reinterpret_cast<unsigned&>(e);
  (void)*reinterpret_cast<unsigned*>(&e);

  // a type that is the signed or unsigned type corresponding a cv-qualified
  // version of the dynamic type the object
  (void)reinterpret_cast<const signed&>(d);
  (void)*reinterpret_cast<const signed*>(&d);
  (void)reinterpret_cast<const signed&>(e);
  (void)*reinterpret_cast<const signed*>(&e);
  (void)reinterpret_cast<const unsigned&>(d);
  (void)*reinterpret_cast<const unsigned*>(&d);
  (void)reinterpret_cast<const unsigned&>(e);
  (void)*reinterpret_cast<const unsigned*>(&e);
  (void)reinterpret_cast<volatile signed&>(d);
  (void)*reinterpret_cast<volatile signed*>(&d);
  (void)reinterpret_cast<volatile signed&>(e);
  (void)*reinterpret_cast<volatile signed*>(&e);
  (void)reinterpret_cast<volatile unsigned&>(d);
  (void)*reinterpret_cast<volatile unsigned*>(&d);
  (void)reinterpret_cast<volatile unsigned&>(e);
  (void)*reinterpret_cast<volatile unsigned*>(&e);
  (void)reinterpret_cast<const volatile signed&>(d);
  (void)*reinterpret_cast<const volatile signed*>(&d);
  (void)reinterpret_cast<const volatile signed&>(e);
  (void)*reinterpret_cast<const volatile signed*>(&e);
  (void)reinterpret_cast<const volatile unsigned&>(d);
  (void)*reinterpret_cast<const volatile unsigned*>(&d);
  (void)reinterpret_cast<const volatile unsigned&>(e);
  (void)*reinterpret_cast<const volatile unsigned*>(&e);

  // an aggregate or union type that includes one of the aforementioned types
  // among its members (including, recursively, a member of a subaggregate or
  // contained union)
  // TODO: checking is not implemented for tag types

  // a type that is a (possible cv-qualified) base class type of the dynamic
  // type of the object
  // TODO: checking is not implemented for tag types

  // a char or unsigned char type
  (void)reinterpret_cast<char&>(a);
  (void)*reinterpret_cast<char*>(&a);
  (void)reinterpret_cast<unsigned char&>(a);
  (void)*reinterpret_cast<unsigned char*>(&a);
  (void)reinterpret_cast<char&>(b);
  (void)*reinterpret_cast<char*>(&b);
  (void)reinterpret_cast<unsigned char&>(b);
  (void)*reinterpret_cast<unsigned char*>(&b);
}

namespace templated {
template <typename TARGETTYPE, typename UATYPE>
void cast_uninstantiated() {
  const UATYPE* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // no warning
}


template <typename TARGETTYPE, typename UATYPE>
void cast_instantiated_badly() {
  const UATYPE* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // expected-warning {{dereference of type 'const int *' that was reinterpret_cast from type 'const float *' has undefined behavior}}
}

template <typename TARGETTYPE, typename UATYPE>
void cast_instantiated_well() {
  const UATYPE* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // no warning
}

template <typename TARGETTYPE>
void cast_one_tmpl_arg_uninstantiated() {
  const int* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // no warning
}

template <typename TARGETTYPE>
void cast_one_tmpl_arg_instantiated_badly() {
  const float* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // expected-warning {{dereference of type 'const int *' that was reinterpret_cast from type 'const float *' has undefined behavior}}
}

template <typename TARGETTYPE>
void cast_one_tmpl_arg_instantiated_well() {
  const float* data;
  (void)*reinterpret_cast<const TARGETTYPE*>(data); // no warning
}

template <int size>
void cast_nontype_template_true_positive_noninstantiated() {
  const float *data;
  const int arr[size];
  (void)*reinterpret_cast<const int*>(data); // expected-warning {{dereference of type 'const int *' that was reinterpret_cast from type 'const float *' has undefined behavior}}
}

template <int size>
void cast_nontype_template_true_negative_noninstantiated() {
  const int data[size];
  (void)*reinterpret_cast<const int*>(data); // no warning
}

void top() {
  cast_instantiated_badly<int, float>();
  // expected-note@-1 {{in instantiation of function template specialization 'templated::cast_instantiated_badly<int, float>' requested here}}
  cast_instantiated_well<int, int>();
  cast_one_tmpl_arg_instantiated_badly<int>();
  // expected-note@-1 {{in instantiation of function template specialization 'templated::cast_one_tmpl_arg_instantiated_badly<int>' requested here}}
  cast_one_tmpl_arg_instantiated_well<float>();
}

template<typename T, typename U>
void cast_template_dependent_type_noninstantiated(T** x)
{
    (void)*reinterpret_cast<U**>(x);
}

template<typename T, typename U>
void cast_template_dependent_member_type_noninstantiated(typename T::X x)
{
    (void)*reinterpret_cast<typename U::Y>(x);
}

} // namespace templated
