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
