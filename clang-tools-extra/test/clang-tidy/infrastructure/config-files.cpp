// RUN: clang-tidy -dump-config %S/Inputs/config-files/- -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}from-parent
// CHECK-BASE: HeaderFilterRegex: parent
// RUN: clang-tidy -dump-config %S/Inputs/config-files/1/- -- | FileCheck %s -check-prefix=CHECK-CHILD1
// CHECK-CHILD1: Checks: {{.*}}from-child1
// CHECK-CHILD1: HeaderFilterRegex: child1
// RUN: clang-tidy -dump-config %S/Inputs/config-files/2/- -- | FileCheck %s -check-prefix=CHECK-CHILD2
// CHECK-CHILD2: Checks: {{.*}}from-parent
// CHECK-CHILD2: HeaderFilterRegex: parent
// RUN: clang-tidy -dump-config %S/Inputs/config-files/3/- -- | FileCheck %s -check-prefix=CHECK-CHILD3
// CHECK-CHILD3: Checks: {{.*}}from-parent,from-child3
// CHECK-CHILD3: HeaderFilterRegex: child3
// RUN: clang-tidy -dump-config -checks='from-command-line' -header-filter='from command line' %S/Inputs/config-files/- -- | FileCheck %s -check-prefix=CHECK-COMMAND-LINE
// CHECK-COMMAND-LINE: Checks: {{.*}}from-parent,from-command-line
// CHECK-COMMAND-LINE: HeaderFilterRegex: from command line

// For this test we have to use names of the real checks because otherwise values are ignored.
// Running with the old key: <Key>, value: <value> CheckOptions
// RUN: clang-tidy -dump-config %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD4
// Running with the new <key>: <value> syntax
// RUN: clang-tidy -dump-config %S/Inputs/config-files/4/key-dict/- -- | FileCheck %s -check-prefix=CHECK-CHILD4

// CHECK-CHILD4: Checks: {{.*}}modernize-loop-convert,modernize-use-using,llvm-qualified-auto
// CHECK-CHILD4-DAG: llvm-qualified-auto.AddConstToQualified: 'true'
// CHECK-CHILD4-DAG: modernize-loop-convert.MaxCopySize: '20'
// CHECK-CHILD4-DAG: modernize-loop-convert.MinConfidence: reasonable
// CHECK-CHILD4-DAG: modernize-use-using.IgnoreMacros: 'false'

// RUN: clang-tidy --explain-config %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-EXPLAIN
// CHECK-EXPLAIN: 'llvm-qualified-auto' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}44{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-loop-convert' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-use-using' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.

// RUN: clang-tidy -dump-config \
// RUN: --config='{InheritParentConfig: true, \
// RUN: Checks: -llvm-qualified-auto, \
// RUN: CheckOptions: [{key: modernize-loop-convert.MaxCopySize, value: 21}]}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD5
// Also test with the {Key: Value} Syntax specified on command line
// RUN: clang-tidy -dump-config \
// RUN: --config='{InheritParentConfig: true, \
// RUN: Checks: -llvm-qualified-auto, \
// RUN: CheckOptions: {modernize-loop-convert.MaxCopySize: 21}}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD5

// CHECK-CHILD5: Checks: {{.*}}modernize-loop-convert,modernize-use-using,llvm-qualified-auto,-llvm-qualified-auto
// CHECK-CHILD5-DAG: modernize-loop-convert.MaxCopySize: '21'
// CHECK-CHILD5-DAG: modernize-loop-convert.MinConfidence: reasonable
// CHECK-CHILD5-DAG: modernize-use-using.IgnoreMacros: 'false'

// RUN: clang-tidy -dump-config \
// RUN: --config='{InheritParentConfig: false, \
// RUN: Checks: -llvm-qualified-auto}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD6
// CHECK-CHILD6: Checks: {{.*-llvm-qualified-auto'? *$}}
// CHECK-CHILD6-NOT: modernize-use-using.IgnoreMacros

// RUN: clang-tidy -dump-config \
// RUN: --config='{CheckOptions: {readability-function-size.LineThreshold: ""}, \
// RUN: Checks: readability-function-size}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD7
// CHECK-CHILD7: readability-function-size.LineThreshold: none


// Validate that check options are printed in alphabetical order:
// RUN: clang-tidy --checks="-*,readability-identifier-naming" --dump-config %S/Inputs/config-files/- -- | grep "readability-identifier-naming\." | sort --check
