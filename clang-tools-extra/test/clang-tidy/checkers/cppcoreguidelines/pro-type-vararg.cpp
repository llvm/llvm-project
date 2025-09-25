// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-vararg %t

void f(int i);
void f_vararg(int i, ...);

struct C {
  void g_vararg(...);
  void g(const char*);
} c;

template<typename... P>
void cpp_vararg(P... p);

void check() {
  f_vararg(1, 7, 9);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not call c-style vararg functions [cppcoreguidelines-pro-type-vararg]
  c.g_vararg("foo");
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not call c-style vararg functions

  f(3); // OK
  c.g("foo"); // OK
  cpp_vararg(1, 7, 9); // OK
}

// ... as a parameter is allowed (e.g. for SFINAE)
template <typename T>
void CallFooIfAvailableImpl(T& t, ...) {
  // nothing
}
template <typename T>
void CallFooIfAvailableImpl(T& t, decltype(t.foo())*) {
  t.foo();
}
template <typename T>
void CallFooIfAvailable(T& t) {
  CallFooIfAvailableImpl(t, 0); // OK to call variadic function when the argument is a literal 0
}

#include <stdarg.h>
void my_printf(const char* format, ...) {
  va_list ap;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare variables of type va_list; use variadic templates instead
  va_start(ap, format);
  va_list n;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare variables of type va_list; use variadic templates instead
  va_copy(n, ap);
  int i = va_arg(ap, int);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not use va_arg to define c-style vararg functions; use variadic templates instead
  va_end(ap);
}

int my_vprintf(const char* format, va_list arg ); // OK to declare function taking va_list

void ignoredBuiltinsTest(void *ptr) {
  (void)__builtin_assume_aligned(ptr, 8);
  (void)__builtin_constant_p(0);
  (void)__builtin_fpclassify(0, 0, 0, 0, 0, 0.f);
  (void)__builtin_isinf_sign(0.f);
  (void)__builtin_prefetch(nullptr);
}

// Some implementations of __builtin_va_list and __builtin_ms_va_list desugared
// as 'char *' or 'void *'. This test checks whether we are handling this case
// correctly and not generating false positives.
void no_false_positive_desugar_va_list(char *in) {
  char *tmp1 = in;
  void *tmp2 = in;
}

namespace PR30542 {
  struct X;
  template <typename T>
  char IsNullConstant(X*);
  template <typename T>
  char (&IsNullConstant(...))[2];

  static_assert(sizeof(IsNullConstant<int>(0)) == 1, "");
  static_assert(sizeof(IsNullConstant<int>(17)) == 2, "");

  using Type = decltype(IsNullConstant<int>(17));
}
