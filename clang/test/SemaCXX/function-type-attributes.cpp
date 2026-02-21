// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace GH142608 {

typedef void (*report_fn)(const char *err);

void die_builtin(const char *err);

__attribute__((noreturn))
report_fn die_routine;

template <class T>
void foo(T, typename T::size = 0); // #foo

void bar() {
	foo<__attribute__((noreturn)) report_fn>(die_routine);
  // expected-error@-1 {{no matching function}}
  // expected-note@#foo {{substitution failure [with T = report_fn]: type 'report_fn' (aka 'void (*)(const char *) __attribute__((noreturn))')}}
}

}
