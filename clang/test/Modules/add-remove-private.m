// RUN: rm -rf %t
// RUN: rm -rf %t.mcp
// RUN: mkdir -p %t
// RUN: cp -r %S/Inputs/AddRemovePrivate.framework %t/AddRemovePrivate.framework

// Build with module.private.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -F %t %s -verify -DP -Wno-private-module
// RUN: cp %t.mcp/AddRemovePrivate.pcm %t/with.pcm

// Build without module.private.modulemap
// RUN: rm %t/AddRemovePrivate.framework/Modules/module.private.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -F %t %s -verify
// RUN: not diff %t.mcp/AddRemovePrivate.pcm %t/with.pcm
// RUN: cp %t.mcp/AddRemovePrivate.pcm %t/without.pcm
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -F %t %s -DP 2>&1 | FileCheck %s
// CHECK: no submodule named 'Private'

// Build with module.private.modulemap (again)
// RUN: cp %S/Inputs/AddRemovePrivate.framework/Modules/module.private.modulemap %t/AddRemovePrivate.framework/Modules/module.private.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -F %t %s -verify -DP -Wno-private-module
// RUN: not diff %t.mcp/AddRemovePrivate.pcm %t/without.pcm

// expected-no-diagnostics

@import AddRemovePrivate;
#ifdef P
@import AddRemovePrivate.Private;
#endif
