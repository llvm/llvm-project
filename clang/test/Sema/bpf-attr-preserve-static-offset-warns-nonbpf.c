// RUN: %clang_cc1 -fsyntax-only -verify %s

#define __pso __attribute__((preserve_static_offset))

struct foo { int a; } __pso; // expected-warning{{unknown attribute}}
union quux { int a; } __pso; // expected-warning{{unknown attribute}}
