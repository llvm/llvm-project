// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++2a -verify %t/M.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoGlobalFrag.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoModuleDecl.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoPrivateFrag.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoModuleDeclAndNoPrivateFrag.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoGlobalFragAndNoPrivateFrag.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoGlobalFragAndNoModuleDecl.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/NoGlobalFragAndNoModuleDeclAndNoPrivateFrag.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/ExportFrags.cppm

//--- M.cppm
module;
extern int a; // #a1
export module Foo;

int a; // expected-error {{declaration of 'a' in module Foo follows declaration in the global module}}
       // expected-note@#a1 {{previous decl}}
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
module :private; // #priv-frag
int b; // ok
module :private; // expected-error {{private module fragment redefined}}
                 // expected-note@#priv-frag {{previous definition is here}}

//--- NoGlobalFrag.cppm

extern int a; // #a1
export module Foo; // expected-error {{module declaration must occur at the start of the translation unit}}
                   // expected-note@-2 {{add 'module;' to the start of the file to introduce a global module fragment}}

// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}

int a; // #a2
extern int b;
module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
module :private; // #priv-frag
int b; // ok
module :private; // expected-error {{private module fragment redefined}}
// expected-note@#priv-frag {{previous definition is here}}

//--- NoModuleDecl.cppm
module; // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
extern int a; // #a1
int a; // #a2
extern int b;
module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
module :private; // expected-error {{private module fragment declaration with no preceding module declaration}}
int b; // ok

//--- NoPrivateFrag.cppm
module;
extern int a; // #a1
export module Foo;

// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}
int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
int b; // ok


//--- NoModuleDeclAndNoPrivateFrag.cppm
module; // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
extern int a; // #a1
int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}

int b; // ok

//--- NoGlobalFragAndNoPrivateFrag.cppm
extern int a; // #a1
export module Foo; // expected-error {{module declaration must occur at the start of the translation unit}}
// expected-note@1 {{add 'module;' to the start of the file to introduce a global module fragment}}

// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}

int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}

int b; // ok

//--- NoGlobalFragAndNoModuleDecl.cppm
extern int a; // #a1
int a; // #a2
extern int b;
module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
module :private; // #priv-frag
// expected-error@-1 {{private module fragment declaration with no preceding module declaration}}
int b; // ok


//--- NoGlobalFragAndNoModuleDeclAndNoPrivateFrag.cppm
extern int a; // #a1
int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}
int b; // ok

//--- ExportFrags.cppm
export module; // expected-error {{global module fragment cannot be exported}}
extern int a; // #a1
export module Foo;
// expected-error@#a2 {{declaration of 'a' in module Foo follows declaration in the global module}}
// expected-note@#a1 {{previous decl}}

int a; // #a2
extern int b;

module; // expected-error {{'module;' introducing a global module fragment can appear only at the start of the translation unit}}

module :private; // #priv-frag

int b; // ok
module :private; // expected-error {{private module fragment redefined}}
                 // expected-note@#priv-frag {{previous definition is here}}
