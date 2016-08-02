// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral -Wformat-non-iso -fblocks %s

#include <stdarg.h>

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
extern int vprintf(const char *restrict, va_list);
}

void f(char **sp, float *fp) {
  scanf("%as", sp); // expected-warning{{'a' length modifier is not supported by ISO C}}

  // TODO: Warn that the 'a' conversion specifier is a C++11 feature.
  printf("%a", 1.0);
  scanf("%afoobar", fp);
}

void g() {
  printf("%ls", "foo"); // expected-warning{{format specifies type 'wchar_t *' but the argument has type 'const char *'}}
}

// Test that we properly handle format_idx on C++ members.
class Foo {
public:
  const char *gettext(const char *fmt) __attribute__((format_arg(2)));

  int scanf(const char *, ...) __attribute__((format(scanf, 2, 3)));
  int printf(const char *, ...) __attribute__((format(printf, 2, 3)));
  int printf2(const char *, ...);

  static const char *gettext_static(const char *fmt) __attribute__((format_arg(1)));
  static int printf_static(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
};

void h(int *i) {
  Foo foo;
  foo.scanf("%d"); // expected-warning{{more '%' conversions than data arguments}}
  foo.printf("%d", i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
  Foo::printf_static("%d", i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}

  printf(foo.gettext("%d"), i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
  printf(Foo::gettext_static("%d"), i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
}

// Test handling __null for format string literal checking.
extern "C" {
  int test_null_format(const char *format, ...) __attribute__((__format__ (__printf__, 1, 2)));
}

void rdar8269537(const char *f)
{
  test_null_format(false); // expected-warning {{null from a constant boolean}}
  test_null_format(0); // no-warning
  test_null_format(__null); // no-warning
  test_null_format(f); // expected-warning {{not a string literal}}
}

int Foo::printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap,fmt);
  const char * const format = fmt;
  vprintf(format, ap); // no-warning

  const char *format2 = fmt;
  vprintf(format2, ap); // expected-warning{{format string is not a string literal}}

  return 0;
}

int Foo::printf2(const char *fmt, ...) {
  va_list ap;
  va_start(ap,fmt);
  vprintf(fmt, ap); // expected-warning{{format string is not a string literal}}

  return 0;
}


namespace Templates {
  template<typename T>
  void my_uninstantiated_print(const T &arg) {
    printf("%d", arg); // no-warning
  }

  template<typename T>
  void my_print(const T &arg) {
    printf("%d", arg); // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
  }

  void use_my_print() {
    my_print("abc"); // expected-note {{requested here}}
  }


  template<typename T>
  class UninstantiatedPrinter {
  public:
    static void print(const T &arg) {
      printf("%d", arg); // no-warning
    }
  };

  template<typename T>
  class Printer {
    void format(const char *fmt, ...) __attribute__((format(printf,2,3)));
  public:

    void print(const T &arg) {
      format("%d", arg); // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
    }
  };

  void use_class(Printer<const char *> &p) {
    p.print("abc"); // expected-note {{requested here}}
  }

  
  extern void (^block_print)(const char * format, ...) __attribute__((format(printf, 1, 2)));

  template<typename T>
  void uninstantiated_call_block_print(const T &arg) {
    block_print("%d", arg); // no-warning
  }

  template<typename T>
  void call_block_print(const T &arg) {
    block_print("%d", arg); // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
  }

  void use_block_print() {
    call_block_print("abc"); // expected-note {{requested here}}
  }
}

namespace implicit_this_tests {
struct t {
    void func1(const char *, ...) __attribute__((__format__(printf, 1, 2))); // expected-error {{format attribute cannot specify the implicit this argument as the format string}}
    void (*func2)(const char *, ...) __attribute__((__format__(printf, 1, 2)));
    static void (*func3)(const char *, ...) __attribute__((__format__(printf, 1, 2)));
    static void func4(const char *, ...) __attribute__((__format__(printf, 1, 2)));
};

void f() {
  t t1;
  t1.func2("Hello %s"); // expected-warning {{more '%' conversions than data arguments}}
  t::func3("Hello %s"); // expected-warning {{more '%' conversions than data arguments}}
  t::func4("Hello %s"); // expected-warning {{more '%' conversions than data arguments}}
}
}
