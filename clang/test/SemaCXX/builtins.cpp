// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11 -fcxx-exceptions
// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++1z -fcxx-exceptions
typedef const struct __CFString * CFStringRef;
#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  (void)CFStringRef(CFSTR("Hello"));
}

void a() { __builtin_va_list x, y; ::__builtin_va_copy(x, y); }

// <rdar://problem/10063539>
template<int (*Compare)(const char *s1, const char *s2)>
int equal(const char *s1, const char *s2) {
  return Compare(s1, s2) == 0;
}
template int equal<&__builtin_strcmp>(const char*, const char*); // expected-error {{builtin functions must be directly called}}

// PR13195
void f2() {
  __builtin_isnan; // expected-error {{builtin functions must be directly called}}
}

// pr14895
typedef __typeof(sizeof(int)) size_t;
extern "C" void *__builtin_alloca (size_t);

namespace addressof {
  struct S {} s;
  static_assert(__builtin_addressof(s) == &s, "");

  struct T { constexpr T *operator&() const { return nullptr; } int n; } t;
  constexpr T *pt = __builtin_addressof(t);
  static_assert(&pt->n == &t.n, "");

  struct U { int n : 5; } u;
  int *pbf = __builtin_addressof(u.n); // expected-error {{address of bit-field requested}}

  S *ptmp = __builtin_addressof(S{}); // expected-error {{taking the address of a temporary}}
}

void no_ms_builtins() {
  __assume(1); // expected-error {{use of undeclared}}
  __noop(1); // expected-error {{use of undeclared}}
  __debugbreak(); // expected-error {{use of undeclared}}
}

struct FILE;
extern "C" int vfprintf(FILE *__restrict, const char *__restrict,
                        __builtin_va_list va);

void synchronize_args() {
  __sync_synchronize(0); // expected-error {{too many arguments}}
}

namespace test_launder {
#define TEST_TYPE(Ptr, Type) \
  static_assert(__is_same(decltype(__builtin_launder(Ptr)), Type), "expected same type")

struct Dummy {};

using FnType = int(char);
using MemFnType = int (Dummy::*)(char);
using ConstMemFnType = int (Dummy::*)() const;

void foo() {}

void test_builtin_launder_diags(void *vp, const void *cvp, FnType *fnp,
                                MemFnType mfp, ConstMemFnType cmfp, int (&Arr)[5]) {
  __builtin_launder(vp);   // expected-error {{void pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(cvp);  // expected-error {{void pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(fnp);  // expected-error {{function pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(mfp);  // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(cmfp); // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  (void)__builtin_launder(&fnp);
  __builtin_launder(42);      // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(nullptr); // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(foo);     // expected-error {{function pointer argument to '__builtin_launder' is not allowed}}
  (void)__builtin_launder(Arr);
}

void test_builtin_launder(char *p, const volatile int *ip, const float *&fp,
                          double *__restrict dp) {
  int x;
  __builtin_launder(x); // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}

  TEST_TYPE(p, char*);
  TEST_TYPE(ip, const volatile int*);
  TEST_TYPE(fp, const float*);
  TEST_TYPE(dp, double *__restrict);

  char *d = __builtin_launder(p);
  const volatile int *id = __builtin_launder(ip);
  int *id2 = __builtin_launder(ip); // expected-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'const volatile int *'}}
  const float* fd = __builtin_launder(fp);
}

void test_launder_return_type(const int (&ArrayRef)[101], int (&MArrRef)[42][13],
                              void (**&FuncPtrRef)()) {
  TEST_TYPE(ArrayRef, const int *);
  TEST_TYPE(MArrRef, int(*)[13]);
  TEST_TYPE(FuncPtrRef, void (**)());
}

template <class Tp>
constexpr Tp *test_constexpr_launder(Tp *tp) {
  return __builtin_launder(tp);
}
constexpr int const_int = 42;
constexpr int const_int2 = 101;
constexpr const int *const_ptr = test_constexpr_launder(&const_int);
static_assert(&const_int == const_ptr, "");
static_assert(const_ptr != test_constexpr_launder(&const_int2), "");

void test_non_constexpr() {
  constexpr int i = 42;                            // expected-note {{declared here}}
  constexpr const int *ip = __builtin_launder(&i); // expected-error {{constexpr variable 'ip' must be initialized by a constant expression}}
  // expected-note@-1 {{pointer to 'i' is not a constant expression}}
}

constexpr bool test_in_constexpr(const int &i) {
  return (__builtin_launder(&i) == &i);
}

static_assert(test_in_constexpr(const_int), "");
void f() {
  constexpr int i = 42;
  static_assert(test_in_constexpr(i), "");
}

struct Incomplete; // expected-note {{forward declaration}}
struct IncompleteMember {
  Incomplete &i;
};
void test_incomplete(Incomplete *i, IncompleteMember *im) {
  // expected-error@+1 {{incomplete type 'test_launder::Incomplete' where a complete type is required}}
  __builtin_launder(i);
  __builtin_launder(&i); // OK
  __builtin_launder(im); // OK
}

void test_noexcept(int *i) {
  static_assert(noexcept(__builtin_launder(i)), "");
}
#undef TEST_TYPE
} // end namespace test_launder

template<typename T> void test_builtin_complex(T v, double d) {
  (void)__builtin_complex(v, d); // expected-error {{different types}} expected-error {{not a real floating}}
  (void)__builtin_complex(d, v); // expected-error {{different types}} expected-error {{not a real floating}}
  (void)__builtin_complex(v, v); // expected-error {{not a real floating}}
}
template void test_builtin_complex(double, double);
template void test_builtin_complex(float, double); // expected-note {{instantiation of}}
template void test_builtin_complex(int, double); // expected-note {{instantiation of}}
