// RUN: %clang_cc1 -fsyntax-only -fno-spell-checking -verify %s
typedef struct {
  float x, y;
} Point;

point p1; // expected-error{{unknown type name 'point'}}
