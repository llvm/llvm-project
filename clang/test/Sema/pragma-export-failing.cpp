// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -x c++ -triple s390x-none-zos -fzos-extensions %s -fsyntax-only -verify

#pragma export(f0(int))                    // expected-warning{{missing ')' after '#pragma export' - ignoring}}
#pragma export(f3(double, double, double)) // expected-warning{{missing ')' after '#pragma export' - ignoring}}

#pragma export(N::sf1(void)) // expected-warning{{missing ')' after '#pragma export' - ignoring}}
#pragma export(N::s1) // expected-warning{{missing ')' after '#pragma export' - ignoring}}
namespace N {
static void sf1(void) {}
static int s1;

static void sf0(void) {}
int v0;
static int s0;
}
#pragma export(N::sf0(void)) // expected-warning{{missing ')' after '#pragma export' - ignoring}}
#pragma export(N::s0) // expected-warning{{missing ')' after '#pragma export' - ignoring}}

void f10(int);
#pragma export(f10) // expected-warning{{failed to resolve '#pragma export' to a declaration}}

#pragma export(f11) // expected-warning{{failed to resolve '#pragma export' to a declaration}}
void f11(int);

template<auto func>
struct S {

#pragma export(func) // expected-error{{this pragma cannot appear in struct declaration}}
};

extern "C" void funcToExport();

S<funcToExport> s{};
