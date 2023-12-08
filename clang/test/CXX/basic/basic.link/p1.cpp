// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_GLOBAL_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_MODULE_DECL %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_PRIVATE_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_MODULE_DECL -DNO_PRIVATE_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_GLOBAL_FRAG -DNO_PRIVATE_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_GLOBAL_FRAG -DNO_MODULE_DECL %s
// RUN: %clang_cc1 -std=c++2a -verify -DNO_GLOBAL_FRAG -DNO_MODULE_DECL -DNO_PRIVATE_FRAG %s
// RUN: %clang_cc1 -std=c++2a -verify -DEXPORT_FRAGS %s

#ifndef NO_GLOBAL_FRAG
#ifdef EXPORT_FRAGS
export // expected-error {{global module fragment cannot be exported}}
#endif
module;
#ifdef NO_MODULE_DECL
// expected-error@-2 {{missing 'module' declaration at end of global module fragment introduced here}}
#endif
#endif

extern int a; // #a1

#ifndef NO_MODULE_DECL
export module Foo;
#ifdef NO_GLOBAL_FRAG
// expected-error@-2 {{module declaration must occur at the start of the translation unit}}
// expected-note@1 {{add 'module;' to the start of the file to introduce a global module fragment}}
#endif

// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}
#endif

int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}

#ifndef NO_PRIVATE_FRAG
#ifdef EXPORT_FRAGS
export // expected-error {{private module fragment cannot be exported}}
#endif
module :private; // #priv-frag
#ifdef NO_MODULE_DECL
// expected-error@-2 {{private module fragment declaration with no preceding module declaration}}
#endif
#endif

int b; // ok


#ifndef NO_PRIVATE_FRAG
#ifndef NO_MODULE_DECL
module :private; // expected-error {{private module fragment redefined}}
// expected-note@#priv-frag {{previous definition is here}}
#endif
#endif
