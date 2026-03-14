// RUN: %clang_cc1 -fsyntax-only -Wformat -verify %s -Wno-error=non-pod-varargs
// RUN: %clang_cc1 -fsyntax-only -Wformat -verify -std=c++98 %s -Wno-error=non-pod-varargs
// RUN: %clang_cc1 -fsyntax-only -Wformat -verify -std=c++11 %s -Wno-error=non-pod-varargs

#include <stdarg.h>

extern "C" {
extern int printf(const char *restrict, ...);
extern int sprintf(char *, const char *restrict, ...);
}

class HasCStr {
  const char *str;
 public:
  HasCStr(const char *s): str(s) { }
  const char *c_str() {return str;}
};

class HasNoCStr {
  const char *str;
 public:
  HasNoCStr(const char *s): str(s) { }
  const char *not_c_str() {return str;}
};

extern const char extstr[16];
void pod_test() {
  char str[] = "test";
  char dest[32];
  char formatString[] = "non-const %s %s";
  HasCStr hcs(str);
  HasNoCStr hncs(str);
  int n = 10;

  printf("%d: %s\n", n, hcs.c_str());
  printf("%d: %s\n", n, hcs);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{cannot pass non-POD object of type 'HasCStr' to variadic function; expected type from format string was 'char *'}}
  // expected-note@-3 {{did you mean to call the c_str() method?}}
#else
  // expected-warning@-5 {{format specifies type 'char *' but the argument has type 'HasCStr'}}
#endif

  printf("%d: %s\n", n, hncs);
#if __cplusplus <= 199711L
 // expected-warning@-2 {{cannot pass non-POD object of type 'HasNoCStr' to variadic function; expected type from format string was 'char *'}}
#else
  // expected-warning@-4 {{format specifies type 'char *' but the argument has type 'HasNoCStr'}}
#endif

  sprintf(str, "%d: %s", n, hcs);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{cannot pass non-POD object of type 'HasCStr' to variadic function; expected type from format string was 'char *'}}
  // expected-note@-3 {{did you mean to call the c_str() method?}}
#else
  // expected-warning@-5 {{format specifies type 'char *' but the argument has type 'HasCStr'}}
#endif

  printf(formatString, hcs, hncs);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{cannot pass object of non-POD type 'HasCStr' through variadic function}}
  // expected-warning@-3 {{cannot pass object of non-POD type 'HasNoCStr' through variadic function}}
#endif

  printf(extstr, hcs, n);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{cannot pass object of non-POD type 'HasCStr' through variadic function}}
#endif
}

struct Printf {
  Printf();
  Printf(const Printf&);
  Printf(const char *,...) __attribute__((__format__(__printf__,2,3)));
};

void constructor_test() {
  const char str[] = "test";
  HasCStr hcs(str);
  Printf p("%s %d %s", str, 10, 10); // expected-warning {{format specifies type 'char *' but the argument has type 'int'}}
  Printf q("%s %d", hcs, 10);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{cannot pass non-POD object of type 'HasCStr' to variadic constructor; expected type from format string was 'char *'}}
  // expected-note@-3 {{did you mean to call the c_str() method?}}
#else
  // expected-warning@-5 {{format specifies type 'char *' but the argument has type 'HasCStr'}}
#endif
}
