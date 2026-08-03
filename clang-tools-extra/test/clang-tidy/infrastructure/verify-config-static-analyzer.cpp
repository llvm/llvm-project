// REQUIRES: static-analyzer
// RUN: echo -e 'Checks: "-*,clang-analyzer-optin.cplusplus.UninitializedObject"\nCheckOptions:\n  clang-analyzer-optin.cplusplus.UninitializedObject:Pedantic: true' > %t.MyClangTidyConfigCSA
// RUN: clang-tidy --verify-config --config-file=%t.MyClangTidyConfigCSA 2>&1 | FileCheck %s -check-prefix=CHECK-VERIFY-CSA-OK -implicit-check-not='{{warnings|error}}'
// CHECK-VERIFY-CSA-OK: No config errors detected.

// RUN: echo -e 'Checks: "-*,clang-analyzer-optin.cplusplus.UninitializedObject"\nCheckOptions:\n  clang-analyzer-optin.cplusplus.UninitializedObject.Pedantic: true' > %t.MyClangTidyConfigCSABad
// RUN: not clang-tidy --verify-config --config-file=%t.MyClangTidyConfigCSABad 2>&1 | FileCheck %s -check-prefix=CHECK-VERIFY-CSA-BAD -implicit-check-not='{{warnings|error}}'
// CHECK-VERIFY-CSA-BAD: command-line option '-config': warning: unknown check option 'clang-analyzer-optin.cplusplus.UninitializedObject.Pedantic'; did you mean 'clang-analyzer-optin.cplusplus.UninitializedObject:Pedantic' [-verify-config]
