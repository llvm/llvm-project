// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x c %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c %t
// RUN: grep "Rectangle" %t
struct Point {
  float x, y;
};

struct Rectangle {
  struct Point top_left, // expected-note{{'top_left' declared here}}
               bottom_right;
};

enum Color { Red, Green, Blue };

struct Window {
  struct Rectangle bounds; // expected-note{{'bounds' declared here}}
  enum Color color;
};

struct Window window = {
  .bunds. // expected-error{{field designator 'bunds' does not refer to any field in type 'struct Window'; did you mean 'bounds'?}}
  topleft.x = 3.14, // expected-error{{field designator 'topleft' does not refer to any field in type 'struct Rectangle'; did you mean 'top_left'?}}
  2.71818, 5.0, 6.0, Red
};

void test() {
  Rectangle r1; // expected-error{{must use 'struct' tag to refer to type 'Rectangle'}}
  r1.top_left.x = 0;

  typedef struct Rectangle Rectangle; // expected-note{{'Rectangle' declared here}}
  rectangle *r2 = &r1; // expected-error{{unknown type name 'rectangle'; did you mean 'Rectangle'?}}
  r2->top_left.y = 0;
  unsinged *ptr = 0; // expected-error{{use of undeclared identifier 'unsinged'; did you mean 'unsigned'?}}
  *ptr = 17;
}
