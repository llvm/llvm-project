// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import TypedefTag;

typedef struct { int x; } TypedefStructHidden_t;
typedef struct { int x; } TypedefStructVisible_t; // expected-error{{typedef redefinition}}
// expected-note@typedef-tag.h:1 {{here}}
