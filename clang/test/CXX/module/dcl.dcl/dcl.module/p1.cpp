// RUN: %clang_cc1 -std=c++20 -verify %s -DFOO=export -DBAR=export
// RUN: %clang_cc1 -std=c++20 -verify %s -DFOO=export -DBAR=
// RUN: %clang_cc1 -std=c++20 %s -DFOO=export -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -fmodule-file=foo=%t -DFOO=
// RUN: %clang_cc1 -std=c++20 %s -fmodule-file=foo=%t -DBAR=export
// RUN: %clang_cc1 -std=c++20 -verify %s -fmodule-file=foo=%t -DFOO= -DBAR=export

#ifdef FOO
FOO module foo; // expected-note {{previous module declaration is here}}
#endif

#ifdef BAR
BAR module bar; // expected-error {{translation unit contains multiple module declarations}}
#endif
