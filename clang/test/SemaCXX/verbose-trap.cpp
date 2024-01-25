// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify %s

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
  __builtin_verbose_trap(""); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a non-empty constant string}}
  __builtin_verbose_trap(); // expected-error {{too few arguments}}
  __builtin_verbose_trap(0); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a non-empty constant string}}
  __builtin_verbose_trap(1); // expected-error {{cannot initialize a parameter of type 'const char *' with}}
  __builtin_verbose_trap(arg); // expected-error {{argument to __builtin_verbose_trap must be a pointer to a non-empty constant string}}
  __builtin_verbose_trap(str);
  __builtin_verbose_trap(u8"hello");
}

void test() {
  f<constMsg3>(nullptr);
}
