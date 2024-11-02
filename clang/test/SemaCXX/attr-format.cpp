// RUN: %clang_cc1 -fsyntax-only -Wformat-nonliteral -verify %s
#include <stdarg.h>

int printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

struct S {
  static void f(const char *, ...) __attribute__((format(printf, 1, 2)));
  static const char *f2(const char *) __attribute__((format_arg(1)));

  // GCC has a hidden 'this' argument in member functions which is why
  // the format argument is argument 2 here.
  void g(const char*, ...) __attribute__((format(printf, 2, 3)));
  const char* g2(const char*) __attribute__((format_arg(2)));

  void h(const char*, ...) __attribute__((format(printf, 1, 4))); // \
      expected-error{{implicit this argument as the format string}}
  void h2(const char*, ...) __attribute__((format(printf, 2, 1))); // \
      expected-error{{out of bounds}}
  const char* h3(const char*) __attribute__((format_arg(1))); // \
      expected-error{{invalid for the implicit this argument}}

  void operator() (const char*, ...) __attribute__((format(printf, 2, 3)));
};

// PR5521
struct A { void a(const char*,...) __attribute((format(printf,2,3))); };
void b(A x) {
  x.a("%d", 3);
}

// PR8625: correctly interpret static member calls as not having an implicit
// 'this' argument.
namespace PR8625 {
  struct S {
    static void f(const char*, const char*, ...)
      __attribute__((format(printf, 2, 3)));
  };
  void test(S s, const char* str) {
    s.f(str, "%s", str);
  }
}

// Make sure we interpret member operator calls as having an implicit
// this argument.
void test_operator_call(S s, const char *str) {
  s("%s", str);
}

template <typename... Args>
void format(const char *fmt, Args &&...args) // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
    __attribute__((format(printf, 1, 2)));

template <typename Arg>
Arg &expand(Arg &a) { return a; }

struct foo {
  int big[10];
  foo();
  ~foo();

  template <typename... Args>
  void format(const char *const fmt, Args &&...args) // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
      __attribute__((format(printf, 2, 3))) {
    printf(fmt, expand(args)...);
  }
};

void format_invalid_nonpod(const char *fmt, struct foo f) // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
    __attribute__((format(printf, 1, 2)));

void do_format() {
  int x = 123;
  int &y = x;
  const char *s = "world";
  format("bare string");
  format("%s", 123); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  format("%s %s %u %d %i %p\n", "hello", s, 10u, x, y, &do_format);
  format("%s %s %u %d %i %p\n", "hello", s, 10u, x, y, do_format);
  format("bad format %s"); // expected-warning{{more '%' conversions than data arguments}}

  struct foo f;
  format_invalid_nonpod("hello %i", f); // expected-warning{{format specifies type 'int' but the argument has type 'struct foo'}}

  f.format("%s", 123); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  f.format("%s %s %u %d %i %p\n", "hello", s, 10u, x, y, &do_format);
  f.format("%s %s %u %d %i %p\n", "hello", s, 10u, x, y, do_format);
  f.format("bad format %s"); // expected-warning{{more '%' conversions than data arguments}}
}
