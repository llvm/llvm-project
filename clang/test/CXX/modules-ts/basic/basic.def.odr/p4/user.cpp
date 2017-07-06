// RUN: %clang_cc1 -fmodules-ts %S/module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s --implicit-check-not=unused --implicit-check-not=global_module

import Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_Z18noninline_exportedv
  noninline_exported();

  // Module-linkage declarations are not visible here.
}
