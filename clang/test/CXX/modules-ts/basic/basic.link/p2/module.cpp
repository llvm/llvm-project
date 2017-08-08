// RUN: %clang_cc1 -std=c++1z -fmodules-ts %S/module.cppm -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -fmodule-file=%t %s -verify
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
