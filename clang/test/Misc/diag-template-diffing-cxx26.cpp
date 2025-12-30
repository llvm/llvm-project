// RUN: %clang_cc1 -fsyntax-only %s -std=c++26                                                  -verify=expected,notree
// RUN: %clang_cc1 -fsyntax-only %s -std=c++26 -fno-elide-type                                  -verify=expected,notree
// RUN: %clang_cc1 -fsyntax-only %s -std=c++26                 -fdiagnostics-show-template-tree -verify=expected,tree
// RUN: %clang_cc1 -fsyntax-only %s -std=c++26 -fno-elide-type -fdiagnostics-show-template-tree -verify=expected,tree

namespace GH93068 {
  int n[2];

  template <auto> struct A {}; // #A

  namespace t1 {
    // notree-error@#1 {{no viable conversion from 'A<0>' to 'A<n + 1>'}}

    /* tree-error@#1 {{no viable conversion
  A<
    [0 != n + 1]>}}*/

    A<n + 1> v1 = A<0>(); // #1
    // expected-note@#A {{no known conversion from 'A<0>' to 'const A<&n[1]> &' for 1st argument}}
    // expected-note@#A {{no known conversion from 'A<0>' to 'A<&n[1]> &&' for 1st argument}}

    // notree-error@#2 {{no viable conversion from 'A<n>' to 'A<n + 1>'}}
    /* tree-error@#2 {{no viable conversion
  A<
    [n != n + 1]>}}*/

    A<n + 1> v2 = A<n>(); // #2
    // expected-note@#A {{no known conversion from 'A<n>' to 'const A<&n[1]> &' for 1st argument}}
    // expected-note@#A {{no known conversion from 'A<n>' to 'A<&n[1]> &&' for 1st argument}}
  } // namespace t1

  namespace t2 {
    A<n> v1;
    A<n + 1> v2;

    // notree-note@#A {{no known conversion from 'A<n>' to 'const A<(no argument)>' for 1st argument}}
    // notree-note@#A {{no known conversion from 'A<n>' to 'A<(no argument)>' for 1st argument}}

    /* tree-note@#A {{no known conversion from argument type to parameter type for 1st argument
  [(no qualifiers) != const] A<
    [n != (no argument)]>}}*/

    /* tree-note@#A {{no known conversion from argument type to parameter type for 1st argument
  A<
    [n != (no argument)]>}}*/

    void f() { v2 = v1; } // expected-error {{no viable overloaded '='}}
  } // namespace t2
} // namespace GH93068
