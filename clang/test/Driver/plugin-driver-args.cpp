/// Test passing args to plugins via the clang driver and -fplugin-arg
// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fplugin-arg-call_super_plugin-help -fsyntax-only -### %s 2>&1 | FileCheck %s

// CHECK: "-load"
// CHECK-SAME: CallSuperAttr
// CHECK-SAME: "-plugin-arg-call_super_plugin"
// CHECK-SAME: "help"

/// Check that dashed-args get forwarded like this to the plugin.
/// Dashes cannot be part of the plugin name here
// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fplugin-arg-call_super_plugin-help-long -fsyntax-only %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CMD
// CHECK-CMD: "-plugin-arg-call_super_plugin" "help-long"

/// Error handling for -fplugin-arg-
// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fplugin-arg- -fsyntax-only %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PLUGIN-NAME
// CHECK-NO-PLUGIN-NAME: missing plugin name in -fplugin-arg-

// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fplugin-arg-testname -fsyntax-only %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PLUGIN-ARG1
// CHECK-NO-PLUGIN-ARG1: missing plugin argument for plugin testname in -fplugin-arg-testname

// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fplugin-arg-testname- -fsyntax-only %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PLUGIN-ARG2
// CHECK-NO-PLUGIN-ARG2: missing plugin argument for plugin testname in -fplugin-arg-testname-
