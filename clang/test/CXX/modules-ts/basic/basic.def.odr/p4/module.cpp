// RUN: %clang_cc1 -fmodules-ts %S/module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s --implicit-check-not=unused --implicit-check-not=global_module

// CHECK-DAG: @extern_var_exported = external global
// CHECK-DAG: @inline_var_exported = linkonce_odr global
// CHECK-DAG: @_ZW6ModuleE19static_var_exported = available_externally global i32 0,
// CHECK-DAG: @const_var_exported = available_externally constant i32 3,
//
// FIXME: The module name should be mangled into all of these.
// CHECK-DAG: @extern_var_module_linkage = external global
// CHECK-DAG: @inline_var_module_linkage = linkonce_odr global
// CHECK-DAG: @_ZW6ModuleE25static_var_module_linkage = available_externally global i32 0,
// CHECK-DAG: @_ZW6ModuleE24const_var_module_linkage = available_externally constant i32 3,

module Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_Z18noninline_exportedv
  noninline_exported();

  (void)&extern_var_exported;
  (void)&inline_var_exported;
  (void)&static_var_exported; // FIXME: Should not be exported.
  (void)&const_var_exported;

  // FIXME: This symbol should not be visible here.
  // CHECK: declare {{.*}}@_ZW6ModuleE26used_static_module_linkagev
  used_static_module_linkage();

  // CHECK: define linkonce_odr {{.*}}@_ZW6ModuleE26used_inline_module_linkagev
  used_inline_module_linkage();

  // CHECK: declare {{.*}}@_ZW6ModuleE24noninline_module_linkagev
  noninline_module_linkage();

  (void)&extern_var_module_linkage;
  (void)&inline_var_module_linkage;
  (void)&static_var_module_linkage; // FIXME: Should not be visible here.
  (void)&const_var_module_linkage;
}
