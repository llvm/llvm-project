// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-none-zos -fzos-extensions %s -fsyntax-only -verify

#pragma export(d0)                         // expected-warning{{failed to resolve '#pragma export' to a declaration}}
#pragma export(f9)                         // expected-warning{{failed to resolve '#pragma export' to a declaration}}

#pragma export(sf1) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf1'}}
#pragma export(s1) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's1'}}
static void sf1(void) {}
static int s1;

static void sf0(void) {}
int v0;
static int s0;
#pragma export(sf0) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf0'}}
#pragma export(s0) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's0'}}

#pragma export(f1) // expected-error {{visibility does not match previous declaration}}
int f1() __attribute__((visibility("hidden")));
int f2() __attribute__((visibility("hidden")));
#pragma export(f2) // expected-error {{visibility does not match previous declaration}}


int hoo() __attribute__((visibility("hidden")));

int foo() { return 4; }
#pragma export(foo)  // expected-warning {{#pragma export can only applied before a symbol is defined}}

int var = 4;
#pragma export(var) // expected-warning {{#pragma export can only applied before a symbol is defined}}

int func() {
#pragma export(local) // expected-error{{'#pragma export' can only appear at file scope}}
  int local;
  int l2;
#pragma export(l2) // expected-error{{'#pragma export' can only appear at file scope}}
  return local+l2;
}

int local = 2;
int l2 =4;

