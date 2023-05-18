// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodules-local-submodule-visibility -I%S/Inputs/no-linkage -fmodule-map-file=%S/Inputs/no-linkage/module.modulemap %s -verify

#include "empty.h"

namespace NS { int n; } // expected-note {{candidate}}
struct Typedef { int n; }; // expected-note {{candidate}}
int AliasDecl; // expected-note {{candidate}}
int UsingDecl; // expected-note {{candidate}}
namespace RealNS = NS; // expected-note {{candidate}}
typedef int Struct; // expected-note {{candidate}}
enum { Variable }; // expected-note {{candidate}}
const int AnotherNS = 0; // expected-note {{candidate}}
const int Enumerator = 0; // expected-note {{candidate}}
static int Overloads; // expected-note {{candidate}}

// expected-note@decls.h:1 {{candidate}}
// expected-note@decls.h:2 {{candidate}}
// expected-note@decls.h:3 {{candidate}}
// expected-note@decls.h:4 {{candidate}}
// expected-note@decls.h:5 {{candidate}}
// expected-note@decls.h:6 {{candidate}}
// expected-note@decls.h:7 {{candidate}}
// expected-note@decls.h:8 {{candidate}}
// expected-note@decls.h:9 {{candidate}}
// expected-note@decls.h:10 {{candidate}}
// expected-note@decls.h:11 {{candidate}}

void use(int);
void use_things() {
  use(Typedef().n);
  use(NS::n);
  use(AliasDecl);
  use(UsingDecl);
  use(RealNS::n);
  use(Struct(0));
  use(Variable);
  use(AnotherNS);
  use(Enumerator);
  use(Overloads);
}

#include "decls.h"

void use_things_again() {
  use(Typedef().n); // expected-error {{ambiguous}}
  use(NS::n); // expected-error {{ambiguous}}
  use(AliasDecl); // expected-error {{ambiguous}}
  use(UsingDecl); // expected-error {{ambiguous}}
  use(RealNS::n); // expected-error {{ambiguous}}
  use(Struct(0)); // expected-error {{ambiguous}}
  use(Variable); // expected-error {{ambiguous}}
  use(AnotherNS); // expected-error {{ambiguous}}
  use(Enumerator); // expected-error {{ambiguous}}
  use(Overloads); // expected-error {{ambiguous}}
}
