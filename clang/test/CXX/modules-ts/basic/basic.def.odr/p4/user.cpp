// RUN: %clang_cc1 -fmodules-ts %S/module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s --implicit-check-not=unused --implicit-check-not=global_module

// CHECK-DAG: @extern_var_exported = external {{(dso_local )?}}global
// CHECK-DAG: @inline_var_exported = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @const_var_exported = available_externally {{(dso_local )?}}constant i32 3

import Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_Z18noninline_exportedv
  noninline_exported();

  (void)&extern_var_exported;
  (void)&inline_var_exported;
  (void)&const_var_exported;

  // Module-linkage declarations are not visible here.
}
