// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify

// [cpp.predefined]/p4: If any of the pre-defined macro names in this
// subclause, or the identifier defined, is the subject of a #define or a
// #undef preprocessing directive, the program is ill-formed.
#undef defined  // expected-error {{'defined' cannot be used as a macro name}}
#undef __DATE__ // expected-warning {{undefining builtin macro}}
