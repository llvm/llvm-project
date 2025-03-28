// REQUIRES: systemz-registered-target
// RUN: not %clang_cc1 -triple s390x-none-zos -fzos-extensions %s -fsyntax-only -verify
__attribute__((visibility("hidden"))) int _Export i; // expected-error {{visibility does not match previous declaration}}
class __attribute__((visibility("hidden"))) _Export C; // expected-error {{visibility does not match previous declaration}}

#pragma export(sf1)
#pragma export(s1)
static void sf1(void) {} // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf1'}}
static int s1; // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's1'}}

static void sf0(void) {} // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf0'}}
int v0;
static int s0; // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's0'}}
#pragma export(sf0)
#pragma export(s0)

typedef int _Export ty;
ty x;
int f(int _Export x);
static int _Export s;
struct S {
  int _Export nonstaticdatamember;
};
void g() {
  int _Export automatic;
}

