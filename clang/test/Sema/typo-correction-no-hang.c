// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR50797
struct a {
  int xxx; // expected-note {{'xxx' declared here}}
};

int g_107;
int g_108;
int g_109;

struct a g_999; // expected-note 4{{'g_999' declared here}}

void b(void) { (g_910.xxx = g_910.xxx); } //expected-error 2{{use of undeclared identifier 'g_910'; did you mean 'g_999'}}

void c(void) { (g_910.xxx = g_910.xxx1); } //expected-error 2{{use of undeclared identifier 'g_910'; did you mean 'g_999'}} \
                                             expected-error {{no member named 'xxx1' in 'struct a'; did you mean 'xxx'}}
