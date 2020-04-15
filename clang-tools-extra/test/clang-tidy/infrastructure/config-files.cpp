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
// RUN: clang-tidy -dump-config %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD4
// CHECK-CHILD4: Checks: {{.*}}modernize-loop-convert,modernize-use-using,llvm-qualified-auto
// CHECK-CHILD4: - key: llvm-qualified-auto.AddConstToQualified
// CHECK-CHILD4-NEXT: value: '1
// CHECK-CHILD4: - key: modernize-loop-convert.MaxCopySize
// CHECK-CHILD4-NEXT: value: '20'
// CHECK-CHILD4: - key: modernize-loop-convert.MinConfidence
// CHECK-CHILD4-NEXT: value: reasonable
// CHECK-CHILD4: - key: modernize-use-using.IgnoreMacros
// CHECK-CHILD4-NEXT: value: '0'

// RUN: clang-tidy --explain-config %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-EXPLAIN
// CHECK-EXPLAIN: 'llvm-qualified-auto' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}44{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-loop-convert' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-use-using' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.
