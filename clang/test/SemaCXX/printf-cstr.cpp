// RUN: %clang_cc1 -fsyntax-only -Wformat -verify %s -Wno-error=non-pod-varargs

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
  printf("%d: %s\n", n, hcs); // expected-warning{{cannot pass non-POD object of type 'HasCStr' to variadic function; expected type from format string was 'char *'}} expected-note{{did you mean to call the c_str() method?}}
  printf("%d: %s\n", n, hncs); // expected-warning{{cannot pass non-POD object of type 'HasNoCStr' to variadic function; expected type from format string was 'char *'}}
  sprintf(str, "%d: %s", n, hcs); // expected-warning{{cannot pass non-POD object of type 'HasCStr' to variadic function; expected type from format string was 'char *'}} expected-note{{did you mean to call the c_str() method?}}

  printf(formatString, hcs, hncs); // expected-warning{{cannot pass object of non-POD type 'HasCStr' through variadic function}} expected-warning{{cannot pass object of non-POD type 'HasNoCStr' through variadic function}}
  printf(extstr, hcs, n); // expected-warning{{cannot pass object of non-POD type 'HasCStr' through variadic function}}
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
  Printf q("%s %d", hcs, 10); // expected-warning {{cannot pass non-POD object of type 'HasCStr' to variadic constructor; expected type from format string was 'char *'}} expected-note{{did you mean to call the c_str() method?}}
}
