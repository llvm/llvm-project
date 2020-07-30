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
// CHECK-CHILD4-DAG: - key: llvm-qualified-auto.AddConstToQualified{{ *[[:space:]] *}}value: 'true'
// CHECK-CHILD4-DAG: - key: modernize-loop-convert.MaxCopySize{{ *[[:space:]] *}}value: '20'
// CHECK-CHILD4-DAG: - key: modernize-loop-convert.MinConfidence{{ *[[:space:]] *}}value: reasonable
// CHECK-CHILD4-DAG: - key: modernize-use-using.IgnoreMacros{{ *[[:space:]] *}}value: 'false'

// RUN: clang-tidy --explain-config %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-EXPLAIN
// CHECK-EXPLAIN: 'llvm-qualified-auto' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}44{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-loop-convert' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.
// CHECK-EXPLAIN: 'modernize-use-using' is enabled in the {{.*}}{{[/\\]}}Inputs{{[/\\]}}config-files{{[/\\]}}4{{[/\\]}}.clang-tidy.

// RUN: clang-tidy -dump-config \
// RUN: --config='{InheritParentConfig: true, \
// RUN: Checks: -llvm-qualified-auto, \
// RUN: CheckOptions: [{key: modernize-loop-convert.MaxCopySize, value: 21}]}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD5
// CHECK-CHILD5: Checks: {{.*}}modernize-loop-convert,modernize-use-using,llvm-qualified-auto,-llvm-qualified-auto
// CHECK-CHILD5-DAG: - key: modernize-loop-convert.MaxCopySize{{ *[[:space:]] *}}value: '21'
// CHECK-CHILD5-DAG: - key: modernize-loop-convert.MinConfidence{{ *[[:space:]] *}}value: reasonable
// CHECK-CHILD5-DAG: - key: modernize-use-using.IgnoreMacros{{ *[[:space:]] *}}value: 'false'

// RUN: clang-tidy -dump-config \
// RUN: --config='{InheritParentConfig: false, \
// RUN: Checks: -llvm-qualified-auto}' \
// RUN: %S/Inputs/config-files/4/44/- -- | FileCheck %s -check-prefix=CHECK-CHILD6
// CHECK-CHILD6: Checks: {{.*-llvm-qualified-auto'? *$}}
// CHECK-CHILD6-NOT: - key: modernize-use-using.IgnoreMacros
