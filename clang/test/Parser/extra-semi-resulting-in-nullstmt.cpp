// RUN: %clang_cc1 -fsyntax-only -Wextra-semi-stmt -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wextra-semi-stmt -fixit %t
// RUN: %clang_cc1 -x c++ -Wextra-semi-stmt -Werror %t

#define GOODMACRO(varname) int varname
#define BETTERMACRO(varname) GOODMACRO(varname);
#define NULLMACRO(varname)

enum MyEnum {
  E1,
  E2
};

void test() {
  ; // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}
  ;

  // This removal of extra semi also consumes all the comments.
  // clang-format: off
  ;;;
  // clang-format: on

  // clang-format: off
  ;NULLMACRO(ZZ);
  // clang-format: on

  {}; // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}

  {
    ; // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}
  }

  if (true) {
    ; // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}
  }

  GOODMACRO(v0); // OK

  GOODMACRO(v1;) // OK

  BETTERMACRO(v2) // OK

  BETTERMACRO(v3;) // Extra ';', but within macro expansion, so ignored.

  BETTERMACRO(v4); // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}

  BETTERMACRO(v5;); // expected-warning {{empty expression statement has no effect; remove unnecessary ';' to silence this warning}}

  NULLMACRO(v6) // OK

  NULLMACRO(v7); // OK. This could be either GOODMACRO() or BETTERMACRO() situation, so we can't know we can drop it.

  if (true)
    ; // OK

  while (true)
    ; // OK

  do
    ; // OK
  while (true);

  for (;;) // OK
    ;      // OK

  MyEnum my_enum;
  switch (my_enum) {
  case E1:
    // stuff
    break;
  case E2:; // OK
  }

  for (;;) {
    for (;;) {
      goto contin_outer;
    }
  contin_outer:; // OK
  }
}

;

namespace NS {};

void foo(int x) {
  switch (x) {
  case 0:
    [[fallthrough]];
  case 1:
    return;
  }
}

[[]];
