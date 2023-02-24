// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/M.cppm -verify
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=M=%t/M.pcm %t/M.cpp -verify
//
// RUN: %clang_cc1 -std=c++20 -fmodule-file=M=%t/M.pcm %t/user.cpp -verify

//--- M.cppm
// expected-no-diagnostics
export module M;

export int external_linkage_var;
int module_linkage_var;
static int internal_linkage_var;

export void external_linkage_fn() {}
void module_linkage_fn() {}
static void internal_linkage_fn() {}

export struct external_linkage_class {};
struct module_linkage_class {};
namespace {
struct internal_linkage_class {};
} // namespace

void use() {
  external_linkage_fn();
  module_linkage_fn();
  internal_linkage_fn();
  (void)external_linkage_class{};
  (void)module_linkage_class{};
  (void)internal_linkage_class{};
  (void)external_linkage_var;
  (void)module_linkage_var;
  (void)internal_linkage_var;
}

//--- M.cpp
// expected-no-diagnostics
module M;

// FIXME: Use of internal linkage entities should be rejected.
void use_from_module_impl() {
  external_linkage_fn();
  module_linkage_fn();
  internal_linkage_fn();
  (void)external_linkage_class{};
  (void)module_linkage_class{};
  (void)internal_linkage_class{};
  (void)external_linkage_var;
  (void)module_linkage_var;
  (void)internal_linkage_var;
}

//--- user.cpp
import M;

void use_from_module_impl() {
  external_linkage_fn();
  module_linkage_fn();   // expected-error {{declaration of 'module_linkage_fn' must be imported}}
  internal_linkage_fn(); // expected-error {{declaration of 'internal_linkage_fn' must be imported}}
  (void)external_linkage_class{};
  (void)module_linkage_class{}; // expected-error {{undeclared identifier}} expected-error 0+{{}}
  (void)internal_linkage_class{}; // expected-error {{undeclared identifier}} expected-error 0+{{}}
  // expected-note@M.cppm:9 {{declaration here is not visible}}
  // expected-note@M.cppm:10 {{declaration here is not visible}}
  (void)external_linkage_var;
  (void)module_linkage_var; // expected-error {{undeclared identifier}}
  (void)internal_linkage_var; // expected-error {{undeclared identifier}}
}
