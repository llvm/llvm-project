// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl > %t.out 2> %t.err
// RUN: FileCheck %s --check-prefix=OUT < %t.out
// RUN: FileCheck %s --check-prefix=ERR --implicit-check-not='fatal error:' < %t.err
// RUN: not clang-repl -Xcc -fms-extensions '__pragma(section("broken", )' \
// RUN:   'extern "C" int puts(const char *); ::puts("recovered");' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MS

// MS: error: missing terminating ')' character
// MS: error: Parsing failed.
// MS: {{^recovered$}}

extern "C" int printf(const char *, ...);

void foo() { int x = 5;
// ERR: error: expected '}'
// ERR: error: Parsing failed.

int g1 = 0; void foo() { g1 = 5; } foo(); printf("g1 = %d\n", ::g1);
// OUT: g1 = 5

void (*test)() = [](){ if }
// ERR: error: expected '(' after 'if'
// ERR: error: Parsing failed.

int g2 = 0; void (*test)() = [](){ if (1) g2 = 7; }; test(); printf("g2 = %d\n", ::g2);
// OUT: g2 = 7

namespace myspace {
// ERR: error: expected '}'
// ERR: error: Parsing failed.

namespace myspace { int v = 11; } printf("v = %d\n", ::myspace::v);
// OUT: v = 11

extern "C" { int cvalue;
// ERR: error: expected '}'
// ERR: error: Parsing failed.

extern "C" { int cvalue = 13; } printf("cvalue = %d\n", ::cvalue);
// OUT: cvalue = 13

struct X { using type = int };
// ERR: error: expected ';' after alias declaration
// ERR: error: Parsing failed.

struct X { using type = int; }; X::type t = 3; printf("t = %d\n", ::t);
// OUT: t = 3

%quit
