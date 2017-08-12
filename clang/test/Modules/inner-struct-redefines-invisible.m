// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -verify %s
// expected-no-diagnostics

@import innerstructredef.one;

struct Outer {
// Should set lexical context when parsing 'Inner' here, otherwise there's a crash:
struct Inner {
  int x;
} field;
};
