// RUN: %clang_cc1 -fmodules-ts %S/module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s --implicit-check-not=unused --implicit-check-not=global_module

// CHECK-DAG: @extern_var_exported = external global
// FIXME: Should this be 'external global'?
// CHECK-DAG: @inline_var_exported = linkonce_odr global
// FIXME: These should be 'extern global' and 'extern constant'.
// CHECK-DAG: @_ZW6ModuleE19static_var_exported = global
// CHECK-DAG: @const_var_exported = constant

import Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_Z18noninline_exportedv
  noninline_exported();

  (void)&extern_var_exported;
  (void)&inline_var_exported;
  (void)&static_var_exported;
  (void)&const_var_exported;

  // Module-linkage declarations are not visible here.
}
