// RUN: rm -rf %t
// Run without global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t/cache -fdisable-module-hash -fmodules -fimplicit-module-maps -fno-modules-global-index -F %S/Inputs %s -verify
// RUN: ls %t/cache | not grep modules.idx
// Run and create the global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t/cache -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: ls %t/cache | grep modules.idx
// Run and use the global module index
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t/cache -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify -print-stats 2>&1 | FileCheck %s

// Skip index generation while another module is being built.
// RUN: rm %t/cache/modules.idx
// RUN: touch %t/cache/building.pcm.lock
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t/cache -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: not test -e %t/cache/modules.idx
// Resume index generation once the module lock is removed.
// RUN: rm %t/cache/building.pcm.lock
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t/cache -fdisable-module-hash -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: test -f %t/cache/modules.idx

// expected-no-diagnostics
@import DependsOnModule;
@import Module;

// CHECK: *** Global Module Index Statistics:

int *get_sub(void) {
  return Module_Sub;
}
