// RUN: %clang -### -fbasic-block-sections=none %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NONE %s
// RUN: %clang -### -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-ALL %s
// RUN: %clang -### -fbasic-block-sections=list=%s %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LIST %s
// RUN: %clang -### -fbasic-block-sections=labels %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LABELS %s
//
// CHECK-OPT-NONE: "-fbasic-block-sections=none"
// CHECK-OPT-ALL: "-fbasic-block-sections=all"
// CHECK-OPT-LIST: "-fbasic-block-sections={{[^ ]*}}fbasic-block-sections.c"
// CHECK-OPT-LABELS: "-fbasic-block-sections=labels"
