// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -x c++ -triple s390x-none-zos -fzos-extensions %s -fsyntax-only -verify

#pragma export(f0(int))                    // expected-warning{{failed to resolve '#pragma export' to a declaration}}
#pragma export(f3(double, double, double)) // expected-warning{{failed to resolve '#pragma export' to a declaration}}

#pragma export(N::sf1(void)) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf1'}}
#pragma export(N::s1) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's1'}}
namespace N {
static void sf1(void) {}
static int s1;

static void sf0(void) {}
int v0;
static int s0;
}
#pragma export(N::sf0(void)) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf0'}}
#pragma export(N::s0) // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's0'}}

void f10(int);
#pragma export(f10) // expected-warning{{failed to resolve '#pragma export' to a declaration}}

#pragma export(f11) // expected-warning{{failed to resolve '#pragma export' to a declaration}}
void f11(int);

