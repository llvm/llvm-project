// RUN: rm -rf %t
// Run without global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t -fdisable-module-hash -fmodules -fimplicit-module-maps -fno-modules-global-index -F %S/Inputs %s -verify
// RUN: ls %t|not grep modules.idx
// Run and create the global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: ls %t|grep modules.idx
// Run and use the global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify -print-stats 2>&1 | FileCheck %s

// expected-no-diagnostics
@import DependsOnModule;
@import Module;

// CHECK: *** Global Module Index Statistics:

int *get_sub(void) {
  return Module_Sub;
}
