// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodules-local-submodule-visibility -I%S/Inputs/no-linkage -fmodule-map-file=%S/Inputs/no-linkage/module.modulemap %s -verify

#include "empty.h"

namespace NS { int n; } // expected-note {{candidate}}
struct Typedef { int n; }; // expected-note {{candidate}}
int AliasDecl; // expected-note {{candidate}}
int UsingDecl; // expected-note {{candidate}}

// expected-note@decls.h:2 {{candidate}}
// expected-note@decls.h:3 {{candidate}}
// expected-note@decls.h:4 {{candidate}}
// expected-note@decls.h:5 {{candidate}}

void use(int);
void use_things() {
  use(Typedef().n);
  use(NS::n);
  use(AliasDecl);
  use(UsingDecl);
}

#include "decls.h"

void use_things_again() {
  use(Typedef().n); // expected-error {{ambiguous}}
  use(NS::n); // expected-error {{ambiguous}}
  use(AliasDecl); // expected-error {{ambiguous}}
  use(UsingDecl); // expected-error {{ambiguous}}
}
