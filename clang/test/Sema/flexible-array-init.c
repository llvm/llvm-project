// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
struct one {
  int a;
  int values[]; // expected-note 4{{initialized flexible array member 'values' is here}}
} x = {5, {1, 2, 3}}; // expected-warning{{flexible array initialization is a GNU extension}}

struct one x2 = { 5, 1, 2, 3 }; // expected-warning{{flexible array initialization is a GNU extension}}

void test(void) {
  struct one x3 = {5, {1, 2, 3}}; // expected-error{{initialization of flexible array member is not allowed}}
  struct one x3a = { 5 };
  struct one x3b = { .a = 5 };
  struct one x3c = { 5, {} }; // expected-warning{{use of an empty initializer is a C2x extension}} \
  // expected-warning{{flexible array initialization is a GNU extension}} \
  // expected-warning{{zero size arrays are an extension}}
}

struct foo { 
  int x; 
  int y[]; // expected-note 8 {{initialized flexible array member 'y' is here}}
}; 
struct bar { struct foo z; }; // expected-warning {{'z' may not be nested in a struct due to flexible array member}}
     
struct foo a = { 1, { 2, 3, 4 } };        // expected-warning{{flexible array initialization is a GNU extension}}
struct bar b = { { 1, { 2, 3, 4 } } };    // expected-error{{initialization of flexible array member is not allowed}}
struct bar c = { { 1, { } } };            // // expected-warning{{flexible array initialization is a GNU extension}} \
              // expected-warning{{use of an empty initializer is a C2x extension}} \
              // expected-warning{{zero size arrays are an extension}}
struct foo d[1] = { { 1, { 2, 3, 4 } } };  // expected-warning{{'struct foo' may not be used as an array element due to flexible array member}} \
              // expected-error{{initialization of flexible array member is not allowed}}

struct foo desig_foo = { .y = {2, 3, 4} }; // expected-warning{{flexible array initialization is a GNU extension}}
struct bar desig_bar = { .z.y = { } }; // expected-warning{{use of an empty initializer is a C2x extension}} \
  // expected-warning{{zero size arrays are an extension}} \
  // expected-warning{{flexible array initialization is a GNU extension}}
struct bar desig_bar2 = { .z.y = { 2, 3, 4} }; // expected-error{{initialization of flexible array member is not allowed}}
struct foo design_foo2 = { .y = 2 }; // expected-error{{flexible array requires brace-enclosed initializer}}

struct point {
  int x, y;
};

struct polygon {
  int numpoints;
  struct point points[]; // expected-note{{initialized flexible array member 'points' is here}}
};
struct polygon poly = { 
  .points[2] = { 1, 2} }; // expected-error{{designator into flexible array member subobject}}

// PR3540
struct X {
  int a;
  int b;
  char data[];
};

struct Y {
  int a:4;
  int b:4;
  int c;
  int d;
  int e;
  struct X xs[]; // expected-warning{{'struct X' may not be used as an array element due to flexible array member}}
};


// PR8217
struct PR8217a {
  int  i;
  char v[]; // expected-note 2 {{initialized flexible array member 'v' is here}}
};

void PR8217(void) {
  struct PR8217a foo1 = { .i = 0, .v = "foo" }; // expected-error {{initialization of flexible array member is not allowed}}
  struct PR8217a foo2 = { .i = 0 };
  struct PR8217a foo3 = { .i = 0, .v = { 'b', 'a', 'r', '\0' } }; // expected-error {{initialization of flexible array member is not allowed}}
  struct PR8217a bar;
}

typedef struct PR10648 {
 unsigned long n;
 int v[]; // expected-note {{initialized flexible array member 'v' is here}}
} PR10648;
int f10648(void) { 
  return (PR10648){2, {3, 4}}.v[1]; // expected-error {{initialization of flexible array member is not allowed}}
}

struct FlexWithUnnamedBitfield { int : 10; int x; int y[]; }; // expected-note {{initialized flexible array member 'y' is here}}
void TestFlexWithUnnamedBitfield(void) {
  struct FlexWithUnnamedBitfield x = {10, {3}}; // expected-error {{initialization of flexible array member is not allowed}}
}
