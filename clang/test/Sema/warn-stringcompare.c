// RUN: %clang_cc1 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cxx %s

#define DELIM "/"
#define DOT "."
#define NULL (void *)0

void test(const char *d) {
  if ("/" != d) // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;
  if (d == "/") // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;
  if ("/" != NULL)
    return;
  if (NULL == "/")
    return;
  if ("/" != DELIM) // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;         // cxx-warning@-1 {{comparison between two arrays}}
  if (DELIM == "/") // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;         // cxx-warning@-1 {{comparison between two arrays}}
  if (DELIM != NULL)
    return;
  if (NULL == DELIM)
    return;
  if (DOT != DELIM) // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;         // cxx-warning@-1 {{comparison between two arrays}}
  if (DELIM == DOT) // expected-warning {{result of comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
    return;         // cxx-warning@-1 {{comparison between two arrays}}
}
