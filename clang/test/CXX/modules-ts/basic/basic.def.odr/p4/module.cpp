// RUN: %clang_cc1 -fmodules-ts %S/module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s --implicit-check-not=unused --implicit-check-not=global_module

module Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_Z18noninline_exportedv
  noninline_exported();

  // FIXME: This symbol should not be visible here.
  // CHECK: define internal {{.*}}@_ZL26used_static_module_linkagev
  used_static_module_linkage();

  // FIXME: The module name should be mangled into the name of this function.
  // CHECK: define linkonce_odr {{.*}}@_Z26used_inline_module_linkagev
  used_inline_module_linkage();

  // FIXME: The module name should be mangled into the name of this function.
  // CHECK: declare {{.*}}@_Z24noninline_module_linkagev
  noninline_module_linkage();
}
