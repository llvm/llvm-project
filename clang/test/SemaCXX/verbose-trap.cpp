// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify %s

#if !__has_builtin(__builtin_verbose_trap)
#error
#endif

constexpr char const* constMsg1 = "hello";
char const* const constMsg2 = "hello";
char const constMsg3[] = "hello";

template <const char * const str>
void f(const char * arg) {
  __builtin_verbose_trap("Arbitrary string literals can be used!");
  __builtin_verbose_trap("Argument_must_not_be_null");
  __builtin_verbose_trap("hello" "world");
  __builtin_verbose_trap(constMsg1);
  __builtin_verbose_trap(constMsg2);
  __builtin_verbose_trap("");
  __builtin_verbose_trap(); // expected-error {{too few arguments}}
  __builtin_verbose_trap(0); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a constant string}}
  __builtin_verbose_trap(1); // expected-error {{cannot initialize a parameter of type 'const char *' with an rvalue of type 'int'}}
  __builtin_verbose_trap(arg); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a constant string}}
  __builtin_verbose_trap(str);
  __builtin_verbose_trap(u8"hello");
#if __cplusplus >= 202002L
  // FIXME: Accept c++20 u8 string literals.
  // expected-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char8_t[6]'}}
#endif
  __builtin_verbose_trap("abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd");
}

void test() {
  f<constMsg3>(nullptr);
}
