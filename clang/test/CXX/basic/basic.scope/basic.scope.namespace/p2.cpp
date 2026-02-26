// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++2a -I%t -emit-module-interface %t/interface.cppm -o %t.pcm
// RUN: %clang_cc1 -std=c++2a -I%t -fmodule-file=A=%t.pcm %t/implA.cppm -verify -fno-modules-error-recovery
// RUN: %clang_cc1 -std=c++2a -I%t -fmodule-file=A=%t.pcm %t/implB.cppm -verify -fno-modules-error-recovery

//--- foo.h
#ifndef FOO_H
#define FOO_H
extern int in_header;
#endif

//--- interface.cppm
module;
#include "foo.h"
// FIXME: The following need to be moved to a header file. The global module
// fragment is only permitted to contain preprocessor directives.
int global_module_fragment;
export module A;
export int exported;
int not_exported;
static int internal;

module :private;
int not_exported_private;
static int internal_private;

//--- implA.cppm
module;

void test_early() {
  in_header = 1; // expected-error {{use of undeclared identifier 'in_header'}}
  // expected-note@* {{not visible}}

  global_module_fragment = 1; // expected-error {{use of undeclared identifier 'global_module_fragment'}}

  exported = 1; // expected-error {{use of undeclared identifier 'exported'}}

  not_exported = 1; // expected-error {{use of undeclared identifier 'not_exported'}}

  // FIXME: We need better diagnostic message for static variable.
  internal = 1; // expected-error {{use of undeclared identifier 'internal'}}

  not_exported_private = 1; // expected-error {{undeclared identifier}}

  internal_private = 1; // expected-error {{undeclared identifier}}
}

module A;

void test_late() {
  in_header = 1; // expected-error {{missing '#include "foo.h"'; 'in_header' must be declared before it is used}}
  // expected-note@* {{not visible}}

  global_module_fragment = 1; // expected-error {{missing '#include'; 'global_module_fragment' must be declared before it is used}}

  exported = 1;

  not_exported = 1;

  internal = 1; // expected-error {{use of undeclared identifier 'internal'}}

  not_exported_private = 1;

  internal_private = 1; // expected-error {{use of undeclared identifier 'internal_private'}}
}

//--- implB.cppm
module;

void test_early() {
  in_header = 1; // expected-error {{use of undeclared identifier 'in_header'}}
  // expected-note@* {{not visible}}

  global_module_fragment = 1; // expected-error {{use of undeclared identifier 'global_module_fragment'}}

  exported = 1; // expected-error {{use of undeclared identifier 'exported'}}

  not_exported = 1; // expected-error {{use of undeclared identifier 'not_exported'}}

  // FIXME: We need better diagnostic message for static variable.
  internal = 1; // expected-error {{use of undeclared identifier 'internal'}}

  not_exported_private = 1; // expected-error {{undeclared identifier}}

  internal_private = 1; // expected-error {{undeclared identifier}}
}

export module B;
import A;

void test_late() {
  in_header = 1; // expected-error {{missing '#include "foo.h"'; 'in_header' must be declared before it is used}}
  // expected-note@* {{not visible}}

  global_module_fragment = 1; // expected-error {{missing '#include'; 'global_module_fragment' must be declared before it is used}}

  exported = 1;

  not_exported = 1; // expected-error {{use of undeclared identifier 'not_exported'; did you mean 'exported'?}}
  // expected-note@* {{'exported' declared here}}

  internal = 1; // expected-error {{use of undeclared identifier 'internal'}}

  not_exported_private = 1;
  // FIXME: should not be visible here
  // expected-error@-2 {{undeclared identifier}}

  internal_private = 1; // expected-error {{use of undeclared identifier 'internal_private'}}
}
