// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

typedef int Object;

struct Object *pp;

Object staticObject1;
