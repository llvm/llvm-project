// RUN: %clang_cc1 -std=c2x -verify %s

/* WG14 N3006: Full
 * Underspecified object declarations
 */

struct S1 { int x, y; };        // expected-note {{previous definition is here}}
union U1 { int a; double b; };  // expected-note {{previous definition is here}}
enum E1 { FOO, BAR };           // expected-note {{previous definition is here}}

auto normal_struct = (struct S1){ 1, 2 };
auto normal_struct2 = (struct S1) { .x = 1, .y = 2 };
auto underspecified_struct = (struct S2 { int x, y; }){ 1, 2 };           // expected-error {{'struct S2' is defined as an underspecified object initializer}}
auto underspecified_struct_redef = (struct S1 { char x, y; }){ 'A', 'B'}; // expected-error {{redefinition of 'S1'}}
auto underspecified_empty_struct = (struct S3 { }){ };                    // expected-error {{'struct S3' is defined as an underspecified object initializer}}

auto normal_union_int = (union U1){ .a = 12 };
auto normal_union_double = (union U1){ .b = 2.4 };
auto underspecified_union = (union U2 { int a; double b; }){ .a = 34 };         // expected-error {{'union U2' is defined as an underspecified object initializer}}
auto underspecified_union_redef = (union U1 { char a; double b; }){ .a = 'A' }; // expected-error {{redefinition of 'U1'}}
auto underspecified_empty_union = (union U3 {  }){  };                          // expected-error {{'union U3' is defined as an underspecified object initializer}}

auto normal_enum_foo = (enum E1){ FOO };
auto normal_enum_bar = (enum E1){ BAR };
auto underspecified_enum = (enum E2 { BAZ, QUX }){ BAZ };       // expected-error {{'enum E2' is defined as an underspecified object initializer}}
auto underspecified_enum_redef = (enum E1 { ONE, TWO }){ ONE }; // expected-error {{redefinition of 'E1'}}
auto underspecified_empty_enum = (enum E3 {  }){ };             // expected-error {{'enum E3' is defined as an underspecified object initializer}} \
                                                                   expected-error {{use of empty enum}}

// Constexpr tests
constexpr auto ce_struct = (struct S1){ 1, 2 };
constexpr auto ce_union = (union U1){ .a = 12 };
constexpr auto ce_enum = (enum E1){ FOO };

int func() {
  struct S { int x, y; };
  constexpr int i = (struct T { int a, b; }){0, 1}.a;

  struct T t = { 1, 2 };
}

void func2() {
  int x = (struct Foo { int x; }){ 0 }.x;
}

void func3() {
  constexpr int x = (struct Foo { int x; }){ 0 }.x;
}

void test() {
    constexpr typeof(struct s *) x = 0; // declares `s` which is not an ordinary identifier
    constexpr struct S { int a, b; } y = { 0 }; // declares `S`, `a`, and `b`, none of which are ordinary identifiers
    constexpr int a = 0, b = 0; // declares `a` and `b` as ordinary identifiers
    auto c = (struct T { int x, y; }){0, 0}; // declares `T`, `x`, and `y`, none of which are ordinary identifiers
    constexpr int (*fp)(struct X { int x; } val) = 0; // declares `X` and `x` which are not ordinary identifiers
}
