// RUN: clang-tidy -verify-config --config='' | FileCheck %s -check-prefix=CHECK-VERIFY-OK
// CHECK-VERIFY-OK: No config errors detected.

// RUN: not clang-tidy -verify-config \
// RUN: --checks='-*,bad*glob,llvm*,llvm-includeorder,my-made-up-check' --config='{Checks: "readability-else-after-ret", \
// RUN: HeaderFileExtensions: ["h", "hh", "hpp"], \
// RUN: ImplementationFileExtensions: ["c", "cc", "hpp"], \
// RUN: CheckOptions: {InvalidOption: true, \
// RUN:                modernize-lop-convert.UseCxx20ReverseRanges: true \
// RUN:               }}' 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-VERIFY -implicit-check-not='{{warning|error}}:'

// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check 'readability-else-after-ret'; did you mean 'readability-else-after-return' [-verify-config]
// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check option 'modernize-lop-convert.UseCxx20ReverseRanges'; did you mean 'modernize-loop-convert.UseCxx20ReverseRanges' [-verify-config]
// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check option 'InvalidOption' [-verify-config]
// CHECK-VERIFY-DAG: command-line option '-config': warning: HeaderFileExtension 'hpp' is the same as ImplementationFileExtension 'hpp' [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: check glob 'bad*glob' doesn't match any known check [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: unknown check 'llvm-includeorder'; did you mean 'llvm-include-order' [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: unknown check 'my-made-up-check' [-verify-config]

// RUN: echo -e 'Checks: |\n bugprone-argument-comment\n bugprone-assert-side-effect,\n bugprone-bool-pointer-implicit-conversion\n readability-use-anyof*' > %t.MyClangTidyConfig
// RUN: clang-tidy -verify-config \
// RUN: --config-file=%t.MyClangTidyConfig | FileCheck %s -check-prefix=CHECK-VERIFY-BLOCK-OK
// CHECK-VERIFY-BLOCK-OK: No config errors detected.

// RUN: echo -e 'Checks: |\n bugprone-arguments-*\n bugprone-assert-side-effects\n bugprone-bool-pointer-implicit-conversion' > %t.MyClangTidyConfigBad
// RUN: not clang-tidy -verify-config \
// RUN: --config-file=%t.MyClangTidyConfigBad 2>&1 | FileCheck %s -check-prefix=CHECK-VERIFY-BLOCK-BAD
// CHECK-VERIFY-BLOCK-BAD: command-line option '-config': warning: check glob 'bugprone-arguments-*' doesn't match any known check [-verify-config]
// CHECK-VERIFY-BLOCK-BAD: command-line option '-config': warning: unknown check 'bugprone-assert-side-effects'; did you mean 'bugprone-assert-side-effect' [-verify-config]

// RUN: echo -e 'Checks: "-*,clang-analyzer-optin.cplusplus.UninitializedObject"\nCheckOptions:\n clang-analyzer-optin.cplusplus.UninitializedObject:Pedantic: true' > %t.MyClangTidyConfigCSA
// RUN: clang-tidy --verify-config --config-file=%t.MyClangTidyConfigCSA 2>&1 | FileCheck %s -check-prefix=CHECK-VERIFY-CSA-OK -implicit-check-not='{{warnings|error}}'
// CHECK-VERIFY-CSA-OK: No config errors detected.

// RUN: echo -e 'Checks: "-*,clang-analyzer-optin.cplusplus.UninitializedObject"\nCheckOptions:\n clang-analyzer-optin.cplusplus.UninitializedObject.Pedantic: true' > %t.MyClangTidyConfigCSABad
// RUN: not clang-tidy --verify-config --config-file=%t.MyClangTidyConfigCSABad 2>&1 | FileCheck %s -check-prefix=CHECK-VERIFY-CSA-BAD -implicit-check-not='{{warnings|error}}'
// CHECK-VERIFY-CSA-BAD: command-line option '-config': warning: unknown check option 'clang-analyzer-optin.cplusplus.UninitializedObject.Pedantic'; did you mean 'clang-analyzer-optin.cplusplus.UninitializedObject:Pedantic' [-verify-config]
