// RUN: %clang_cc1 -std=c23 -verify %s

/* WG14 N3006: Yes
 * Underspecified object declarations
 */

void struct_test(void) {
  struct S1 { int x, y; };                                                  // expected-note {{field 'x' has type 'int' here}}

  auto normal_struct = (struct S1){ 1, 2 };
  auto normal_struct2 = (struct S1) { .x = 1, .y = 2 };
  auto underspecified_struct = (struct S2 { int x, y; }){ 1, 2 };
  auto underspecified_struct_redef = (struct S1 { char x, y; }){ 'A', 'B'}; // expected-error {{type 'struct S1' has incompatible definitions}} \
                                                                               expected-error {{cannot use 'auto' with array in C}} \
                                                                               expected-note {{field 'x' has type 'char' here}}
  auto underspecified_empty_struct = (struct S3 { }){ };
  auto zero_init_struct = (struct S4 { int x; }){ 0 };
  int field_struct = (struct S5 { int y; }){ 0 }.y;
}

void union_test(void) {
  union U1 { int a; double b; };                                                  // expected-note {{field 'a' has type 'int' here}}

  auto normal_union_int = (union U1){ .a = 12 };
  auto normal_union_double = (union U1){ .b = 2.4 };
  auto underspecified_union = (union U2 { int a; double b; }){ .a = 34 };
  auto underspecified_union_redef = (union U1 { char a; double b; }){ .a = 'A' }; // expected-error {{type 'union U1' has incompatible definitions}} \
                                                                                     expected-error {{cannot use 'auto' with array in C}} \
                                                                                     expected-note {{field 'a' has type 'char' here}}
  auto underspecified_empty_union = (union U3 {  }){  };
}

void enum_test(void) {
  enum E1 { FOO, BAR };                                           // expected-note {{enumerator 'BAR' with value 1 here}}

  auto normal_enum_foo = (enum E1){ FOO };
  auto normal_enum_bar = (enum E1){ BAR };
  auto underspecified_enum = (enum E2 { BAZ, QUX }){ BAZ };
  auto underspecified_enum_redef = (enum E1 { ONE, TWO }){ ONE }; // expected-error {{type 'enum E1' has incompatible definitions}} \
                                                                     expected-error {{cannot use 'auto' with array in C}} \
                                                                     expected-note {{enumerator 'ONE' with value 0 here}}
  auto underspecified_empty_enum = (enum E3 {  }){ };             // expected-error {{use of empty enum}}
  auto underspecified_undeclared_enum = (enum E4){ FOO };         // expected-error {{variable has incomplete type 'enum E4'}} \
                                                                     expected-note {{forward declaration of 'enum E4'}}
}

void constexpr_test(void) {
  constexpr auto ce_struct = (struct S1){ 1, 2 };                   // expected-error {{variable has incomplete type 'struct S1'}} \
                                                                       expected-note {{forward declaration of 'struct S1'}}
  constexpr auto ce_struct_zero_init = (struct S2 { int x; }){ 0 };
  constexpr int ce_struct_field = (struct S3 { int y; }){ 0 }.y;
  constexpr auto ce_union = (union U1){ .a = 12 };                  // expected-error {{variable has incomplete type 'union U1'}} \
                                                                       expected-note {{forward declaration of 'union U1'}}

  constexpr auto ce_enum = (enum E1 { BAZ, QUX }){ BAZ };
  constexpr auto ce_empty_enum = (enum E2){ FOO };                  // expected-error {{use of undeclared identifier 'FOO'}}
}

void self_reference_test(void) {
  constexpr int i = i;  // expected-error {{constexpr variable 'i' must be initialized by a constant expression}} \
                           expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  auto j = j;           // expected-error {{variable 'j' declared with deduced type 'auto' cannot appear in its own initializer}}
}

void redefinition_test(void) {
  const struct S { int x; } s;  // expected-warning {{default initialization of an object of type 'const struct S' leaves the object uninitialized}} \
                                   expected-note {{previous definition is here}}
  constexpr struct S s = {0};   // expected-error {{redefinition of 's'}}
}

void declaring_an_underspecified_defied_object_test(void) {
  struct S { int x, y; };
  constexpr int i = (struct T { int a, b; }){0, 1}.a;

  struct T t = { 1, 2 };
}

void constexpr_complience_test(void) {
  int x = (struct Foo { int x; }){ 0 }.x;
  constexpr int y = (struct Bar { int x; }){ 0 }.x;
}

void builtin_functions_test(void) {
  constexpr typeof(struct s *) x = 0;
  auto so = sizeof(struct S {});
  auto to = (typeof(struct S {})){};
}

void misc_test(void) {
  constexpr struct S { int a, b; } y = { 0 };
  constexpr int a = 0, b = 0;
  auto c = (struct T { int x, y; }){0, 0};
  auto s2 = ({struct T { int x; } s = {}; s.x; });
  auto s3 = ((struct {}){},0); // expected-warning {{left operand of comma operator has no effect}}
  constexpr int (*fp)(struct X { int x; } val) = 0;
  auto v = (void (*)(int y))0;
}

void misc_struct_test(void) {
  constexpr struct {
      int a;
  } a = {};

  constexpr struct {
      int b;
  } b = (struct S { int x; }){ 0 };  // expected-error-re {{initializing 'const struct (unnamed struct at {{.*}}n3006.c:104:13)' with an expression of incompatible type 'struct S'}}

  auto z = ({
      int a = 12;
      struct {} s;
      a;
  });
}
