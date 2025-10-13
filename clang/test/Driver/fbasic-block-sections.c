// RUN: %clang -### --target=x86_64 -fbasic-block-sections=none %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NONE %s
// RUN: %clang -### --target=x86_64 -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-ALL %s
// RUN: %clang -### --target=x86_64 -fbasic-block-sections=list=%s %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LIST %s
// RUN: %clang -### --target=x86_64 -fbasic-block-sections=labels %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LABELS %s
// RUN: %clang -### --target=aarch64 -fbasic-block-sections=none %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NONE %s
// RUN: %clang -### --target=aarch64 -fbasic-block-sections=list=%s %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LIST %s
// RUN: %clang -### --target=aarch64 -fbasic-block-sections=labels %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-LABELS %s
// RUN: not %clang -### --target=aarch64 -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-INVALID-VALUE %s
// RUN: not %clang -c --target=arm-unknown-linux -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s
// RUN: %clang -### --target=arm-unknown-linux -fbasic-block-sections=all -fbasic-block-sections=none %s -S 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOOPT %s
// RUN: not %clang -c --target=x86_64-apple-darwin10 -fbasic-block-sections=all %s -S 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s
// RUN: not %clang -### --target=x86_64 -fbasic-block-sections=alll %s -S 2>&1 | FileCheck -check-prefix=CHECK-INVALID-VALUE %s
// RUN: not %clang -### --target=x86_64 -fbasic-block-sections=list %s -S 2>&1 | FileCheck -check-prefix=CHECK-INVALID-VALUE %s
// RUN: %clang -### --target=x86_64 -fbasic-block-sections=list= %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NULL-LIST %s
// RUN: %clang -### --target=x86_64 -fbasic-block-sections=none %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT-NONE %s
// RUN: %clang -### -x cuda -nocudainc -nocudalib --target=x86_64 -fbasic-block-sections=all --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s -c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-CUDA %s

//
// CHECK-NOOPT-NOT:  -fbasic-block-sections=
// CHECK-OPT-NONE:   "-fbasic-block-sections=none"
// CHECK-OPT-ALL:    "-fbasic-block-sections=all"
// CHECK-OPT-LIST:   "-fbasic-block-sections={{[^ ]*}}fbasic-block-sections.c"
// CHECK-OPT-LABELS: warning: argument '-fbasic-block-sections=labels' is deprecated, use '-fbasic-block-address-map' instead
// CHECK-OPT-LABELS: "-fbasic-block-address-map"
// CHECK-TRIPLE:     error: unsupported option '-fbasic-block-sections=all' for target
// CHECK-INVALID-VALUE: error: invalid value {{[^ ]*}} in '-fbasic-block-sections={{.*}}'
// CHECK-OPT-NULL-LIST: "-fbasic-block-sections=list="

// GPU-side compilations should have no -fbasic-block-sections. It should only
// be passed to the host compilation
// CHECK-CUDA-NOT:  -fbasic-block-sections=
// CHECK-CUDA: "-cc1" "-triple" "x86_64"
// CHECK-CUDA-SAME: "-fbasic-block-sections=all"
