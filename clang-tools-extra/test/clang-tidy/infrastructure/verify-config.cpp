// RUN: clang-tidy -verify-config --config='' | FileCheck %s -check-prefix=CHECK-VERIFY-OK
// CHECK-VERIFY-OK: No config errors detected.

// RUN: not clang-tidy -verify-config \
// RUN: --checks='-*,bad*glob,llvm*,llvm-includeorder,my-made-up-check' --config='{Checks: "readability-else-after-ret", \
// RUN: CheckOptions: [{key: "IgnoreMacros", value: "true"}, \
// RUN:                {key: "StriceMode", value: "true"}, \
// RUN:                {key: modernize-lop-convert.UseCxx20ReverseRanges, value: true} \
// RUN:               ]}' 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-VERIFY -implicit-check-not='{{warning|error}}:'

// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check 'readability-else-after-ret'; did you mean 'readability-else-after-return' [-verify-config]
// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check option 'modernize-lop-convert.UseCxx20ReverseRanges'; did you mean 'modernize-loop-convert.UseCxx20ReverseRanges' [-verify-config]
// CHECK-VERIFY-DAG: command-line option '-config': warning: unknown check option 'StriceMode'; did you mean 'StrictMode' [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: check glob 'bad*glob' doesn't match any known check [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: unknown check 'llvm-includeorder'; did you mean 'llvm-include-order' [-verify-config]
// CHECK-VERIFY: command-line option '-checks': warning: unknown check 'my-made-up-check' [-verify-config]
