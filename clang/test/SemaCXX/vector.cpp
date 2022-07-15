// RUN: %clang_cc1 -flax-vector-conversions=all -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
// RUN: %clang_cc1 -flax-vector-conversions=all -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -flax-vector-conversions=all -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -flax-vector-conversions=integer -triple x86_64-apple-darwin10 -fsyntax-only -verify %s -DNO_LAX_FLOAT
// RUN: %clang_cc1 -flax-vector-conversions=none -triple x86_64-apple-darwin10 -fsyntax-only -verify %s -DNO_LAX_FLOAT -DNO_LAX_INT

typedef char char16 __attribute__ ((__vector_size__ (16)));
typedef long long longlong16 __attribute__ ((__vector_size__ (16)));
typedef char char16_e __attribute__ ((__ext_vector_type__ (16)));
typedef long long longlong16_e __attribute__ ((__ext_vector_type__ (2)));

// Test overloading and function calls with vector types.
void f0(char16); // expected-note 0+{{candidate}}

void f0_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  f0(c16);
  f0(ll16);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
  f0(c16e);
  f0(ll16e);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
}

int &f1(char16);
float &f1(longlong16);

void f1_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  int &ir1 = f1(c16);
  float &fr1 = f1(ll16);
  int &ir2 = f1(c16e);
  float &fr2 = f1(ll16e);
}

void f2(char16_e); // expected-note 0+{{candidate}}

void f2_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  f2(c16);
  f2(ll16);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
  f2(c16e);
  f2(ll16e); // expected-error{{no matching function}}
  f2('a');
  f2(17);
}

// Test the conditional operator with vector types.
void conditional(bool Cond, char16 c16, longlong16 ll16, char16_e c16e,
                 longlong16_e ll16e) {
  // Conditional operators with the same type.
  __typeof__(Cond? c16 : c16) *c16p1 = &c16;
  __typeof__(Cond? ll16 : ll16) *ll16p1 = &ll16;
  __typeof__(Cond? c16e : c16e) *c16ep1 = &c16e;
  __typeof__(Cond? ll16e : ll16e) *ll16ep1 = &ll16e;

  // Conditional operators with similar types.
  __typeof__(Cond? c16 : c16e) *c16ep2 = &c16e;
  __typeof__(Cond? c16e : c16) *c16ep3 = &c16e;
  __typeof__(Cond? ll16 : ll16e) *ll16ep2 = &ll16e;
  __typeof__(Cond? ll16e : ll16) *ll16ep3 = &ll16e;

  // Conditional operators with compatible types under -flax-vector-conversions (default)
  (void)(Cond? c16 : ll16);
  (void)(Cond? ll16e : c16e);
  (void)(Cond? ll16e : c16);
#ifdef NO_LAX_INT
  // expected-error@-4 {{cannot convert}}
  // expected-error@-4 {{cannot convert}}
  // expected-error@-4 {{cannot convert}}
#endif
}

// Test C++ cast'ing of vector types.
void casts(longlong16 ll16, longlong16_e ll16e) {
  // C-style casts.
  (void)(char16)ll16;
  (void)(char16_e)ll16;
  (void)(longlong16)ll16;
  (void)(longlong16_e)ll16;
  (void)(char16)ll16e;
  (void)(char16_e)ll16e;
  (void)(longlong16)ll16e;
  (void)(longlong16_e)ll16e;

  // Function-style casts.
  (void)char16(ll16);
  (void)char16_e(ll16);
  (void)longlong16(ll16);
  (void)longlong16_e(ll16);
  (void)char16(ll16e);
  (void)char16_e(ll16e);
  (void)longlong16(ll16e);
  (void)longlong16_e(ll16e);

  // static_cast
  (void)static_cast<char16>(ll16);
  (void)static_cast<char16_e>(ll16);
#ifdef NO_LAX_INT
  // expected-error@-3 {{not allowed}}
  // expected-error@-3 {{not allowed}}
#endif
  (void)static_cast<longlong16>(ll16);
  (void)static_cast<longlong16_e>(ll16);
  (void)static_cast<char16>(ll16e);
#ifdef NO_LAX_INT
  // expected-error@-2 {{not allowed}}
#endif
  (void)static_cast<char16_e>(ll16e); // expected-error{{static_cast from 'longlong16_e' (vector of 2 'long long' values) to 'char16_e' (vector of 16 'char' values) is not allowed}}
  (void)static_cast<longlong16>(ll16e);
  (void)static_cast<longlong16_e>(ll16e);

  // reinterpret_cast
  (void)reinterpret_cast<char16>(ll16);
  (void)reinterpret_cast<char16_e>(ll16);
  (void)reinterpret_cast<longlong16>(ll16);
  (void)reinterpret_cast<longlong16_e>(ll16);
  (void)reinterpret_cast<char16>(ll16e);
  (void)reinterpret_cast<char16_e>(ll16e);
  (void)reinterpret_cast<longlong16>(ll16e);
  (void)reinterpret_cast<longlong16_e>(ll16e);
}

template<typename T>
struct convertible_to { // expected-note 3 {{candidate function (the implicit copy assignment operator) not viable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 3 {{candidate function (the implicit move assignment operator) not viable}}
#endif
  operator T() const;
};

void test_implicit_conversions(bool Cond, char16 c16, longlong16 ll16,
                               char16_e c16e, longlong16_e ll16e,
                               convertible_to<char16> to_c16,
                               convertible_to<longlong16> to_ll16,
                               convertible_to<char16_e> to_c16e,
                               convertible_to<longlong16_e> to_ll16e,
                               convertible_to<char16&> rto_c16,
                               convertible_to<char16_e&> rto_c16e) {
  f0(to_c16);
  f0(to_ll16);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
  f0(to_c16e);
  f0(to_ll16e);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
  f2(to_c16);
  f2(to_ll16);
#ifdef NO_LAX_INT
  // expected-error@-2 {{no matching function}}
#endif
  f2(to_c16e);
  f2(to_ll16e); // expected-error{{no matching function}}

  (void)(c16 == c16e);
  (void)(c16 == to_c16);
  (void)+to_c16;
  (void)-to_c16;
  (void)~to_c16;
  (void)(to_c16 == to_c16e);
  (void)(to_c16 != to_c16e);
  (void)(to_c16 <  to_c16e);
  (void)(to_c16 <= to_c16e);
  (void)(to_c16 >  to_c16e);
  (void)(to_c16 >= to_c16e);
  (void)(to_c16 + to_c16);
  (void)(to_c16 - to_c16);
  (void)(to_c16 * to_c16);
  (void)(to_c16 / to_c16);
  (void)(rto_c16 = to_c16); // expected-error{{no viable overloaded '='}}
  (void)(rto_c16 += to_c16);
  (void)(rto_c16 -= to_c16);
  (void)(rto_c16 *= to_c16);
  (void)(rto_c16 /= to_c16);

  (void)+to_c16e;
  (void)-to_c16e;
  (void)~to_c16e;
  (void)(to_c16e == to_c16e);
  (void)(to_c16e != to_c16e);
  (void)(to_c16e <  to_c16e);
  (void)(to_c16e <= to_c16e);
  (void)(to_c16e >  to_c16e);
  (void)(to_c16e >= to_c16e);
  (void)(to_c16e + to_c16);
  (void)(to_c16e - to_c16);
  (void)(to_c16e * to_c16);
  (void)(to_c16e / to_c16);
  (void)(rto_c16e = to_c16); // expected-error{{no viable overloaded '='}}
  (void)(rto_c16e += to_c16);
  (void)(rto_c16e -= to_c16);
  (void)(rto_c16e *= to_c16);
  (void)(rto_c16e /= to_c16);

  (void)+to_c16;
  (void)-to_c16;
  (void)~to_c16;
  (void)(to_c16 == to_c16e);
  (void)(to_c16 != to_c16e);
  (void)(to_c16 <  to_c16e);
  (void)(to_c16 <= to_c16e);
  (void)(to_c16 >  to_c16e);
  (void)(to_c16 >= to_c16e);
  (void)(to_c16 + to_c16e);
  (void)(to_c16 - to_c16e);
  (void)(to_c16 * to_c16e);
  (void)(to_c16 / to_c16e);
  (void)(rto_c16 = c16e); // expected-error{{no viable overloaded '='}}
  (void)(rto_c16 += to_c16e);
  (void)(rto_c16 -= to_c16e);
  (void)(rto_c16 *= to_c16e);
  (void)(rto_c16 /= to_c16e);

  (void)(Cond? to_c16 : to_c16e);
  (void)(Cond? to_ll16e : to_ll16);

  // These 2 are convertible with -flax-vector-conversions (default)
  (void)(Cond? to_c16 : to_ll16);
  (void)(Cond? to_c16e : to_ll16e);
#ifdef NO_LAX_INT
  // expected-error@-3 {{cannot convert}}
  // expected-error@-3 {{cannot convert}}
#endif
}

typedef float fltx2 __attribute__((__vector_size__(8)));
typedef float fltx4 __attribute__((__vector_size__(16)));
typedef double dblx2 __attribute__((__vector_size__(16)));
typedef double dblx4 __attribute__((__vector_size__(32)));

void accept_fltx2(fltx2); // expected-note{{candidate function not viable: no known conversion from 'double' to 'fltx2' (vector of 2 'float' values) for 1st argument}}
void accept_fltx4(fltx4);
void accept_dblx2(dblx2);
#ifdef NO_LAX_FLOAT
// expected-note@-3 {{no known conversion}}
// expected-note@-3 {{no known conversion}}
#endif
void accept_dblx4(dblx4);
void accept_bool(bool); // expected-note{{candidate function not viable: no known conversion from 'fltx2' (vector of 2 'float' values) to 'bool' for 1st argument}}

void test(fltx2 fltx2_val, fltx4 fltx4_val, dblx2 dblx2_val, dblx4 dblx4_val) {
  // Exact matches
  accept_fltx2(fltx2_val);
  accept_fltx4(fltx4_val);
  accept_dblx2(dblx2_val);
  accept_dblx4(dblx4_val);

  // Same-size conversions
  accept_fltx4(dblx2_val);
  accept_dblx2(fltx4_val);
#ifdef NO_LAX_FLOAT
  // expected-error@-3 {{no matching function}}
  // expected-error@-3 {{no matching function}}
#endif

  // Conversion to bool.
  accept_bool(fltx2_val); // expected-error{{no matching function for call to 'accept_bool'}}

  // Scalar-to-vector conversions.
  accept_fltx2(1.0); // expected-error{{no matching function for call to 'accept_fltx2'}}
}

typedef int intx4 __attribute__((__vector_size__(16)));
typedef int inte4 __attribute__((__ext_vector_type__(4)));
typedef float flte4 __attribute__((__ext_vector_type__(4)));

void test_mixed_vector_types(fltx4 f, intx4 n, flte4 g, inte4 m) {
  (void)(f == g);
  (void)(g != f);
  (void)(f <= g);
  (void)(g >= f);
  (void)(f < g);
  (void)(g > f);

  (void)(+g);
  (void)(-g);

  (void)(f + g);
  (void)(f - g);
  (void)(f * g);
  (void)(f / g);
  (void)(f = g);
  (void)(f += g);
  (void)(f -= g);
  (void)(f *= g);
  (void)(f /= g);


  (void)(n == m);
  (void)(m != n);
  (void)(n <= m);
  (void)(m >= n);
  (void)(n < m);
  (void)(m > n);

  (void)(+m);
  (void)(-m);
  (void)(~m);

  (void)(n + m);
  (void)(n - m);
  (void)(n * m);
  (void)(n / m);
  (void)(n % m);
  (void)(n = m);
  (void)(n += m);
  (void)(n -= m);
  (void)(n *= m);
  (void)(n /= m);
}

template<typename T> void test_pseudo_dtor_tmpl(T *ptr) {
  ptr->~T();
  (*ptr).~T();
}

void test_pseudo_dtor(fltx4 *f) {
  f->~fltx4();
  (*f).~fltx4();
  test_pseudo_dtor_tmpl(f);
}

// PR16204
typedef __attribute__((ext_vector_type(4))) int vi4;
const int &reference_to_vec_element = vi4(1).x;

// PR12649
typedef bool bad __attribute__((__vector_size__(16)));  // expected-error {{invalid vector element type 'bool'}}

namespace Templates {
template <typename Elt, unsigned long long Size>
struct TemplateVectorType {
  typedef Elt __attribute__((__vector_size__(Size))) type; // #1
};

template <int N, typename T>
struct PR15730 {
  typedef T __attribute__((vector_size(N * sizeof(T)))) type;
  typedef T __attribute__((vector_size(0x1000000000))) type2; // #2
  typedef T __attribute__((vector_size(3))) type3; // #3
};

void Init() {
  const TemplateVectorType<float, 32>::type Works = {};
  const TemplateVectorType<int, 32>::type Works2 = {};
  // expected-error@#1 {{invalid vector element type 'bool'}}
  // expected-note@+1 {{in instantiation of template class 'Templates::TemplateVectorType<bool, 32>' requested here}}
  const TemplateVectorType<bool, 32>::type NoBool = {};
  // expected-error@#1 {{invalid vector element type 'int __attribute__((ext_vector_type(4)))' (vector of 4 'int' values)}}
  // expected-note@+1 {{in instantiation of template class 'Templates::TemplateVectorType<int __attribute__((ext_vector_type(4))), 32>' requested here}}
  const TemplateVectorType<vi4, 32>::type NoComplex = {};
  // expected-error@#1 {{vector size not an integral multiple of component size}}
  // expected-note@+1 {{in instantiation of template class 'Templates::TemplateVectorType<int, 33>' requested here}}
  const TemplateVectorType<int, 33>::type BadSize = {};
  const TemplateVectorType<int, 3200>::type Large = {};
  // expected-error@#1 {{vector size too large}}
  // expected-note@+1 {{in instantiation of template class 'Templates::TemplateVectorType<int, 68719476736>' requested here}}
  const TemplateVectorType<int, 0x1000000000>::type TooLarge = {};
  // expected-error@#1 {{zero vector size}}
  // expected-note@+1 {{in instantiation of template class 'Templates::TemplateVectorType<int, 0>' requested here}}
  const TemplateVectorType<int, 0>::type Zero = {};

  // expected-error@#2 {{vector size too large}}
  // expected-error@#3 {{vector size not an integral multiple of component size}}
  // expected-note@+1 {{in instantiation of template class 'Templates::PR15730<8, int>' requested here}}
  const PR15730<8, int>::type PR15730_1 = {};
  // expected-error@#2 {{vector size too large}}
  // expected-note@+1 {{in instantiation of template class 'Templates::PR15730<8, char>' requested here}}
  const PR15730<8, char>::type2 PR15730_2 = {};
}

} // namespace Templates

typedef int inte2 __attribute__((__ext_vector_type__(2)));

void test_vector_literal(inte4 res) {
  inte2 a = (inte2)(1, 2); //expected-warning{{left operand of comma operator has no effect}}
  inte4 b = (inte4)(a, a); //expected-error{{C-style cast from vector 'inte2' (vector of 2 'int' values) to vector 'inte4' (vector of 4 'int' values) of different size}} //expected-warning{{left operand of comma operator has no effect}}
}

typedef __attribute__((__ext_vector_type__(4))) float vector_float4;
typedef __attribute__((__ext_vector_type__(4))) int vector_int4;

namespace swizzle_template_confusion {
  template<typename T> struct xyzw {};
  vector_int4 foo123(vector_float4 &A, vector_float4 &B) {
    return A.xyzw < B.x && B.y > A.y; // OK, not a template-id
  }
}

namespace swizzle_typo_correction {
  template<typename T> struct xyzv {};
  vector_int4 foo123(vector_float4 &A, vector_float4 &B) {
    return A.xyzw < B.x && B.y > A.y; // OK, not a typo for 'xyzv'
  }
}

namespace PR45299 {
typedef float float4 __attribute__((vector_size(16)));

// In this example, 'k' is value dependent. PR45299 reported that this asserted
// because of that, since the truncation check attempted to constant evaluate k,
// which it could not do because it is dependent.
template <typename T>
struct NormalMember {
  float4 f(float4 x) {
    return k * x;
  }
  float k;
};

#if __cplusplus >= 201103L
// This should not diagnose, since the constant evaluator (during instantiation)
// can tell that this isn't a truncation.
template <typename T>
struct ConstantValueNoDiag {
  float4 f(float4 x) {
    return k * x;
  }
  static constexpr double k = 1;
};
template <typename T, int N>
struct ConstantValueNoDiagDependentValue {
  float4 f(float4 x) {
    return k * x;
  }
  static constexpr double k = N;
};

// The following two both diagnose because they cause a truncation.  Test both
// the dependent type and non-dependent type versions.
template <typename T>
struct DiagTrunc {
  float4 f(float4 x) {
    // expected-error@+1{{as implicit conversion would cause truncation}}
    return k * x;
  }
  static constexpr double k = 1340282346638528859811704183484516925443.000000;
};
template <typename T, int N>
struct DiagTruncDependentValue {
  float4 f(float4 x) {
    // expected-error@+1{{as implicit conversion would cause truncation}}
    return k * x;
  }
  static constexpr double k = N + 1340282346638528859811704183484516925443.000000;
};
template <typename T>
struct DiagTruncDependentType {
  float4 f(float4 x) {
    // expected-error@+1{{as implicit conversion would cause truncation}}
    return k * x;
  }
  static constexpr T k = 1340282346638528859811704183484516925443.000000;
};

template <typename T>
struct PR45298 {
    T k1 = T(0);
};

// Ensure this no longer asserts.
template <typename T>
struct PR45298Consumer {
  float4 f(float4 x) {
    return (float)s.k1 * x;
  }

  PR45298<T> s;
};
#endif // __cplusplus >= 201103L

void use() {
  float4 theFloat4;
  NormalMember<double>().f(theFloat4);
#if __cplusplus >= 201103L
  ConstantValueNoDiag<double>().f(theFloat4);
  ConstantValueNoDiagDependentValue<double, 1>().f(theFloat4);
  DiagTrunc<double>().f(theFloat4);
  // expected-note@+1{{in instantiation of member function}}
  DiagTruncDependentValue<double, 0>().f(theFloat4);
  // expected-note@+1{{in instantiation of member function}}
  DiagTruncDependentType<double>().f(theFloat4);
  PR45298Consumer<double>().f(theFloat4);
#endif // __cplusplus >= 201103L
}
}

namespace rdar60092165 {
template <class T> void f() {
  typedef T first_type __attribute__((vector_size(sizeof(T) * 4)));
  typedef T second_type __attribute__((vector_size(sizeof(T) * 4)));

  second_type st;
}
}

namespace PR45780 {
enum E { Value = 15 };
void use(char16 c) {
  E e;
  c &Value;   // expected-error{{cannot convert between scalar type 'PR45780::E' and vector type 'char16'}}
  c == Value; // expected-error{{cannot convert between scalar type 'PR45780::E' and vector type 'char16'}}
  e | c;      // expected-error{{cannot convert between scalar type 'E' and vector type 'char16'}}
  e != c;     // expected-error{{cannot convert between scalar type 'E' and vector type 'char16'}}
}

} // namespace PR45780

namespace PR48540 {
// The below used to cause an OOM error, or an assert, make sure it is still
//  valid.
int (__attribute__((vector_size(16))) a);

template <typename T, int I>
struct S {
  T (__attribute__((vector_size(16))) a);
  int (__attribute__((vector_size(I))) b);
  T (__attribute__((vector_size(I))) c);
};

void use() {
  S<int, 16> s;
}
} // namespace PR48540
