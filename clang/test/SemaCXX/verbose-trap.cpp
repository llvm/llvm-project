// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify %s

#if !__has_builtin(__builtin_verbose_trap)
#error
#endif

constexpr char const* constCat1 = "cat";
char const* const constCat2 = "cat";
char const constCat3[] = "cat";

constexpr char const* constMsg1 = "hello";
char const* const constMsg2 = "hello";
char const constMsg3[] = "hello";

template <const char * const category, const char * const reason>
void f(const char * arg) {
  __builtin_verbose_trap("cat1", "Arbitrary string literals can be used!");
  __builtin_verbose_trap(" cat1 ", "Argument_must_not_be_null");
  __builtin_verbose_trap("cat" "egory1", "hello" "world");
  __builtin_verbose_trap(constCat1, constMsg1);
  __builtin_verbose_trap(constCat2, constMsg2);
  __builtin_verbose_trap("", "");
  __builtin_verbose_trap(); // expected-error {{too few arguments}}
  __builtin_verbose_trap(""); // expected-error {{too few arguments}}
  __builtin_verbose_trap("", "", ""); // expected-error {{too many arguments}}
  __builtin_verbose_trap("", 0); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a constant string}}
  __builtin_verbose_trap(1, ""); // expected-error {{cannot initialize a parameter of type 'const char *' with an rvalue of type 'int'}}
  __builtin_verbose_trap(arg, ""); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a constant string}}
  __builtin_verbose_trap("cat$1", "hel$lo"); // expected-error 2 {{argument to __builtin_verbose_trap must not contain $}}
  __builtin_verbose_trap(category, reason);
  __builtin_verbose_trap(u8"cat1", u8"hello");
#if __cplusplus >= 202002L
  // FIXME: Accept c++20 u8 string literals.
  // expected-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char8_t[5]'}}
#endif
  __builtin_verbose_trap("", "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd");
}

template <const char * const category>
void f2() {
  __builtin_verbose_trap(category, 1); // expected-error {{cannot initialize a parameter of type 'const char *' with an rvalue of type 'int'}}
}

void test() {
  f<constCat3, constMsg3>(nullptr);
}
