// Tests that a plugin diagnostic given a stable ID reports it as its SARIF
// ruleId. The stable ID is independent of registration order, unlike the
// numeric ID SARIF falls back to, so a tool keying on diagnostic identity stays
// stable across runs.

// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls \
// RUN:   -fdiagnostics-format sarif %s 2>&1 | FileCheck %s

// REQUIRES: plugins, examples

void f();

// The stable ID the plugin passed to getCustomPluginDiagID is the rule id.
// CHECK: "ruleId": "print_fns_suspicious_decl"
