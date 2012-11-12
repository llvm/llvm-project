// RUN: %clang_cc1 -fsyntax-only -verify -Wreserved-user-defined-literal -fms-extensions -fms-compatibility %s

#define bar(x) #x
const char * f() {
  return "foo"bar("bar")"baz";    /*expected-warning {{identifier after literal will be treated as a reserved user-defined literal suffix in C++11}} */
}
