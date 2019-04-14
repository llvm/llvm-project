// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_GLOBAL_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_MODULE_DECL %s
// RUN: %clang_cc1 -std=c++2a -verify -DEXPORT_FRAGS %s

#ifdef NO_GLOBAL_FRAG
// expected-error@#mod-decl {{module declaration must occur at the start of the translation unit}}
// expected-note@1 {{add 'module;' to the start of the file to introduce a global module fragment}}
#else
#ifdef EXPORT_FRAGS
export // expected-error {{global module fragment cannot be exported}}
#endif
module; // #glob-frag
#endif

extern int a; // #a1

#ifdef NO_MODULE_DECL
// expected-error@#glob-frag {{missing 'module' declaration at end of global module fragment introduced here}}
#else
export module Foo; // #mod-decl

// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}
#endif

int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}

#ifdef EXPORT_FRAGS
export // expected-error {{private module fragment cannot be exported}}
#endif
module :private;

int b; // ok
