// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -DEARLY_IMPORT
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -UEARLY_IMPORT
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -DEARLY_IMPORT -fno-modules-hide-internal-linkage
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -UEARLY_IMPORT -fno-modules-hide-internal-linkage
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -DEARLY_IMPORT -fmodules-local-submodule-visibility
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -UEARLY_IMPORT -fmodules-local-submodule-visibility

#ifdef EARLY_IMPORT
@import using_decl.a;
namespace UsingDecl {
  using ::merged;
}
int k = UsingDecl::merged;
#endif

namespace Y {
  int conflicting_hidden_using_decl;
  int conflicting_hidden_using_decl_fn_2();
  int conflicting_hidden_using_decl_var_2;
  struct conflicting_hidden_using_decl_struct_2;

  struct conflicting_hidden_using_decl_mixed_4 {};
  int conflicting_hidden_using_decl_mixed_5;
  int conflicting_hidden_using_decl_mixed_6();
}

using Y::conflicting_hidden_using_decl;
int conflicting_hidden_using_decl_fn();
int conflicting_hidden_using_decl_var;
struct conflicting_hidden_using_decl_struct {};
using Y::conflicting_hidden_using_decl_fn_2;
using Y::conflicting_hidden_using_decl_var_2;
using Y::conflicting_hidden_using_decl_struct_2;

struct conflicting_hidden_using_decl_mixed_1 {};
int conflicting_hidden_using_decl_mixed_2;
int conflicting_hidden_using_decl_mixed_3();
using Y::conflicting_hidden_using_decl_mixed_4;
using Y::conflicting_hidden_using_decl_mixed_5;
using Y::conflicting_hidden_using_decl_mixed_6;

template<typename T> int use(T);
void test_conflicting() {
  use(conflicting_hidden_using_decl);
  use(conflicting_hidden_using_decl_fn());
  use(conflicting_hidden_using_decl_var);
  use(conflicting_hidden_using_decl_fn_2());
  use(conflicting_hidden_using_decl_var_2);
  use(conflicting_hidden_using_decl_mixed_1());
  use(conflicting_hidden_using_decl_mixed_2);
  use(conflicting_hidden_using_decl_mixed_3());
  use(conflicting_hidden_using_decl_mixed_4());
  use(conflicting_hidden_using_decl_mixed_5);
  use(conflicting_hidden_using_decl_mixed_6());
}

#ifndef EARLY_IMPORT
@import using_decl.a;
#endif

UsingDecl::using_decl_type x = UsingDecl::using_decl_var;
UsingDecl::inner y = x;

@import using_decl.b;

void test_conflicting_2() {
  use(conflicting_hidden_using_decl);         // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_fn());    // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_var);     // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_fn_2());  // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_var_2);   // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_mixed_1); // ok, struct hidden
  use(conflicting_hidden_using_decl_mixed_2); // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_mixed_3); // ok, struct hidden
  use(conflicting_hidden_using_decl_mixed_4); // ok, struct hidden
  use(conflicting_hidden_using_decl_mixed_5); // expected-error {{ambiguous}}
  use(conflicting_hidden_using_decl_mixed_6); // ok, struct hidden
  // expected-note@using-decl.cpp:* 7{{candidate}}
  // expected-note@using-decl-b.h:* 7{{candidate}}

  int conflicting_hidden_using_decl_mixed_1::*p1;
  int conflicting_hidden_using_decl_mixed_3::*p3;
  int conflicting_hidden_using_decl_mixed_4::*p4;
  int conflicting_hidden_using_decl_mixed_6::*p6;
}
