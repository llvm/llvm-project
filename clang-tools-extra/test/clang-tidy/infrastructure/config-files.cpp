// RUN: clang-tidy -dump-config %S/Inputs/config-files/- -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}from-parent
// CHECK-BASE: HeaderFilterRegex: parent
// RUN: clang-tidy -dump-config %S/Inputs/config-files/1/- -- | FileCheck %s -check-prefix=CHECK-CHILD1
// CHECK-CHILD1: Checks: {{.*}}from-child1
// CHECK-CHILD1: HeaderFilterRegex: child1
// RUN: clang-tidy -dump-config %S/Inputs/config-files/2/- -- | FileCheck %s -check-prefix=CHECK-CHILD2
// CHECK-CHILD2: Checks: {{.*}}from-parent
// CHECK-CHILD2: HeaderFilterRegex: parent
// RUN: clang-tidy -dump-config -checks='from-command-line' -header-filter='from command line' %S/Inputs/config-files/- -- | FileCheck %s -check-prefix=CHECK-COMMAND-LINE
// CHECK-COMMAND-LINE: Checks: {{.*}}from-parent,from-command-line
// CHECK-COMMAND-LINE: HeaderFilterRegex: from command line
