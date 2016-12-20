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
// FIXME: Our error recovery here sucks
template int equal<&__builtin_strcmp>(const char*, const char*); // expected-error {{builtin functions must be directly called}} expected-error {{expected unqualified-id}} expected-error {{expected ')'}} expected-note {{to match this '('}}

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
