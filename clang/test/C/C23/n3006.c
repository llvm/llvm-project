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
void constexpr_test() {
  constexpr auto ce_struct = (struct S1){ 1, 2 };
  constexpr auto ce_union = (union U1){ .a = 12 };
  constexpr auto ce_enum = (enum E1){ FOO };
}

void trivial_test() {
  constexpr int i = i;  // expected-error {{constexpr variable 'i' must be initialized by a constant expression}} \
                           expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  auto j = j;           // expected-error {{variable 'j' declared with deduced type 'auto' cannot appear in its own initializer}}
}

void double_definition_test() {
  const struct S { int x; } s;  // expected-note {{previous definition is here}}
  constexpr struct S s = {0};   // expected-error {{redefinition of 's'}}
}

void declaring_an_underspecified_defied_object_test() {
  struct S { int x, y; };
  constexpr int i = (struct T { int a, b; }){0, 1}.a;  // expected-error {{'struct T' is defined as an underspecified object initializer}} \
                                                          FIXME: `constexpr variable 'i' must be initialized by a constant expression` shoud appear

  struct T t = { 1, 2 };                               // TODO: Should this be diagnosed as an invalid declaration?
}

void constexpr_complience_test() {
  int x = (struct Foo { int x; }){ 0 }.x;           // expected-error {{'struct Foo' is defined as an underspecified object initializer}}
  constexpr int y = (struct Bar { int x; }){ 0 }.x; // expected-error {{'struct Bar' is defined as an underspecified object initializer}}
}

void special_test() {
  constexpr typeof(struct s *) x = 0;               // FIXME: declares `s` which is not an ordinary identifier
  constexpr struct S { int a, b; } y = { 0 };    // FIXME: declares `S`, `a`, and `b`, none of which are ordinary identifiers
  constexpr int a = 0, b = 0;
  auto c = (struct T { int x, y; }){0, 0};          // expected-error {{'struct T' is defined as an underspecified object initializer}}
  constexpr int (*fp)(struct X { int x; } val) = 0; // expected-warning {{declaration of 'struct X' will not be visible outside of this function}} \
                                                       FIXME: declares `X` and `x` which are not ordinary identifiers
}
