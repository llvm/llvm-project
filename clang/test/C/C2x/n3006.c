// RUN: %clang_cc1 -std=c2x -verify %s

/* WG14 N3006: Full
 * Underspecified object declarations
 */

struct S1 { int x, y; };        // expected-note {{previous definition is here}}
union U1 { int a; double b; };  // expected-note {{previous definition is here}}
enum E1 { FOO, BAR };           // expected-note {{previous definition is here}}

auto normal_struct = (struct S1){ 1, 2 };
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
