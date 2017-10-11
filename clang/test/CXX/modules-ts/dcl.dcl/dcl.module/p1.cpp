// RUN: %clang_cc1 -std=c++17 -fmodules-ts -verify %s -DFOO=export -DBAR=export
// RUN: %clang_cc1 -std=c++17 -fmodules-ts -verify %s -DFOO=export -DBAR=
// RUN: %clang_cc1 -std=c++17 -fmodules-ts %s -DFOO=export -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++17 -fmodules-ts %s -fmodule-file=%t -DFOO=
// RUN: %clang_cc1 -std=c++17 -fmodules-ts %s -fmodule-file=%t -DBAR=export
// RUN: %clang_cc1 -std=c++17 -fmodules-ts -verify %s -fmodule-file=%t -DFOO= -DBAR=export

#ifdef FOO
FOO module foo; // expected-note {{previous module declaration is here}}
#endif

#ifdef BAR
BAR module bar; // expected-error {{translation unit contains multiple module declarations}}
#endif
