// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -verify=expected,space -Wcompound-token-split

// Ensure we get the same warnings after -frewrite-includes
// RUN: %clang_cc1 %s -E -frewrite-includes -o %t
// RUN: %clang_cc1 -x c++ %t -verify=expected,space -Wcompound-token-split

#ifdef LSQUARE
[
#else

#define VAR(type, name, init) type name = (init)

void f() {
  VAR(int, x, {}); // #1
  // expected-warning@#1 {{'(' and '{' tokens introducing statement expression appear in different macro expansion contexts}}
  // expected-note-re@#1 {{{{^}}'{' token is here}}
  //
  // FIXME: It would be nice to suppress this when we already warned about the opening '({'.
  // expected-warning@#1 {{'}' and ')' tokens terminating statement expression appear in different macro expansion contexts}}
  // expected-note-re@#1 {{{{^}}')' token is here}}
  //
  // expected-error@#1 {{cannot initialize a variable of type 'int' with an rvalue of type 'void'}}
}

#define RPAREN )

int f2() {
  int n = ({ 1; }RPAREN; // expected-warning {{'}' and ')' tokens terminating statement expression appear in different macro expansion contexts}} expected-note {{')' token is here}}
  return n;
}

[ // space-warning-re {{{{^}}'[' tokens introducing attribute are separated by whitespace}}
#define LSQUARE
#include __FILE__
  noreturn ]]  void g();

[[noreturn] ] void h(); // space-warning-re {{{{^}}']' tokens terminating attribute are separated by whitespace}}

struct X {};
int X:: *p; // space-warning {{'::' and '*' tokens forming pointer to member type are separated by whitespace}}

#endif
