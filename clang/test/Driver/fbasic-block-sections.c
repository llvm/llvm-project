// RUN: %clang -### -target x86_64 -fbasic-block-sections=none %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NONE %s
// RUN: %clang -### -target x86_64 -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-ALL %s
// RUN: %clang -### -target x86_64 -fbasic-block-sections=list=%s %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LIST %s
// RUN: %clang -### -target x86_64 -fbasic-block-sections=labels %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LABELS %s
// RUN: not %clang -c -target arm-unknown-linux -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s
// RUN: not %clang -c -target x86_64-apple-darwin10 -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s
//
// CHECK-OPT-NONE:   "-fbasic-block-sections=none"
// CHECK-OPT-ALL:    "-fbasic-block-sections=all"
// CHECK-OPT-LIST:   "-fbasic-block-sections={{[^ ]*}}fbasic-block-sections.c"
// CHECK-OPT-LABELS: "-fbasic-block-sections=labels"
// CHECK-TRIPLE:     error: unsupported option '-fbasic-block-sections=all' for target
