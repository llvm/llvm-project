// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -DgetModuleVersion="epic fail" %s 2>&1 | FileCheck %s

@import DependsOnModule;

// CHECK: While building module 'DependsOnModule' imported from
// CHECK: While building module 'Module' imported from
// CHECK: error: expected ';' after top level declarator
// CHECK: note: expanded from macro 'getModuleVersion'
// CHECK: fatal error: could not build module 'Module'
// CHECK: fatal error: could not build module 'DependsOnModule'
// CHECK-NOT: error:

// RUN: not %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -fdiagnostics-show-note-include-stack 2>&1 | FileCheck -check-prefix=CHECK-REDEF %s
extern int Module;

// CHECK-REDEF: In module 'DependsOnModule' imported from
// CHECK-REDEF: In module 'Module' imported from
// CHECK-REDEF: Module.h:15:12: note: previous definition is here

// RUN: not %clang_cc1 -Wincomplete-umbrella -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -DgetModuleVersion="epic fail" -serialize-diagnostic-file %t/tmp.diag %s 2>&1
// RUN: c-index-test -read-diagnostics %t/tmp.diag 2>&1 | FileCheck -check-prefix=CHECK-SDIAG %s

// CHECK-SDIAG: Module.h:9:13: error: expected ';' after top level declarator
// CHECK-SDIAG: build-fail-notes.m:4:9: note: while building module 'DependsOnModule' imported from
// CHECK-SDIAG: DependsOnModule.h:1:10: note: while building module 'Module' imported from
// CHECK-SDIAG: note: expanded from macro 'getModuleVersion'
// CHECK-SDIAG: warning: umbrella header for module 'Module' does not include header 'NotInModule.h' [-Wincomplete-umbrella]
// CHECK-SDIAG: DependsOnModule.h:1:10: fatal: could not build module 'Module'
// CHECK-SDIAG: build-fail-notes.m:4:9: note: while building module 'DependsOnModule' imported from

