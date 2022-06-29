// RUN: %clang_cc1 -std=c++1z -fmodules-ts %S/module.cppm -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -fmodule-file=%t %s -verify
import M;

void use_from_module_impl() {
  external_linkage_fn();
  module_linkage_fn(); // expected-error {{undeclared identifier}}
  internal_linkage_fn(); // expected-error {{undeclared identifier}}
  (void)external_linkage_class{};
  (void)module_linkage_class{}; // expected-error {{undeclared identifier}} expected-error 0+{{}}
  (void)internal_linkage_class{}; // expected-error {{undeclared identifier}} expected-error 0+{{}}
  // expected-note@module.cppm:9 {{here}}
  (void)external_linkage_var;
  (void)module_linkage_var; // expected-error {{undeclared identifier}}
  (void)internal_linkage_var; // expected-error {{undeclared identifier}}
}
