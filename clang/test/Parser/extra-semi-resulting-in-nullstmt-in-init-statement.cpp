// RUN: %clang_cc1 -fsyntax-only -Wextra -std=c++2a -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi-stmt -std=c++2a -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wempty-init-stmt -std=c++2a -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wempty-init-stmt -std=c++2a -fixit %t
// RUN: %clang_cc1 -x c++ -Wempty-init-stmt -std=c++2a -Werror %t

struct S {
  int *begin();
  int *end();
};

void naive(int x) {
  if (; true) // expected-warning {{empty initialization statement of 'if' has no effect}}
    ;

  switch (; x) { // expected-warning {{empty initialization statement of 'switch' has no effect}}
  }

  for (; int y : S()) // expected-warning {{empty initialization statement of 'range-based for' has no effect}}
    ;

  for (;;) // OK
    ;
}

#define NULLMACRO

void with_null_macro(int x) {
  if (NULLMACRO; true)
    ;

  switch (NULLMACRO; x) {
  }

  for (NULLMACRO; int y : S())
    ;
}

#define SEMIMACRO ;

void with_semi_macro(int x) {
  if (SEMIMACRO true)
    ;

  switch (SEMIMACRO x) {
  }

  for (SEMIMACRO int y : S())
    ;
}

#define PASSTHROUGHMACRO(x) x

void with_passthrough_macro(int x) {
  if (PASSTHROUGHMACRO(;) true)
    ;

  switch (PASSTHROUGHMACRO(;) x) {
  }

  for (PASSTHROUGHMACRO(;) int y : S())
    ;
}
