// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodules-cache-path=%t -fmodule-name=diag_pragma -x c++ %S/Inputs/module.map -fmodules-ts
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_pragma -x c++ %S/Inputs/module.map -fmodules-ts -o %t/explicit.pcm -Werror=string-plus-int
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DEXPLICIT_FLAG -fmodule-file=%t/explicit.pcm

import diag_pragma;

int foo(int x) {
  // Diagnostics from templates in the module follow the diagnostic state from
  // when the module was built.
#ifdef EXPLICIT_FLAG
  // expected-error@diag_pragma.h:7 {{adding 'int' to a string}}
#else
  // expected-warning@diag_pragma.h:7 {{adding 'int' to a string}}
#endif
  // expected-note@diag_pragma.h:7 {{use array indexing}}
  f(0); // expected-note {{instantiation of}}

  g(0); // ok, warning was ignored when building module

  // Diagnostics from this source file ignore the diagnostic state from the
  // module.
  void("foo" + x); // expected-warning {{adding 'int' to a string}}
  // expected-note@-1 {{use array indexing}}

#pragma clang diagnostic ignored "-Wstring-plus-int"

  // Diagnostics from the module ignore diagnostic state changes from this
  // source file.
#ifdef EXPLICIT_FLAG
  // expected-error@diag_pragma.h:7 {{adding 'long' to a string}}
#else
  // expected-warning@diag_pragma.h:7 {{adding 'long' to a string}}
#endif
  // expected-note@diag_pragma.h:7 {{use array indexing}}
  f(0L); // expected-note {{instantiation of}}

  g(0L);

  void("bar" + x);

  if (x = DIAG_PRAGMA_MACRO) // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                             // expected-note {{place parentheses}} expected-note {{use '=='}}
    return 0;
  return 1;
}
