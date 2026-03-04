// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);

void foo() { int x = 5;
// expected-error {{expected '}'}}

int g1 = 0; void foo() { g1 = 5; } foo(); printf("g1 = %d\n", g1);
// CHECK: g1 = 5

void (*test)() = [](){ if }
// expected-error {{expected '(' after 'if'}}

int g2 = 0; void (*test)() = [](){ if (1) g2 = 7; }; test(); printf("g2 = %d\n", g2);
// CHECK: g2 = 7

namespace myspace {
// expected-error {{expected '}'}}

namespace myspace { int v = 11; } printf("v = %d\n", myspace::v);
// CHECK: v = 11

struct X { using type = int };
// expected-error {{expected ';' after alias declaration}}

struct X { using type = int; }; X::type t = 3; printf("t = %d\n", t);
// CHECK: t = 3

%quit