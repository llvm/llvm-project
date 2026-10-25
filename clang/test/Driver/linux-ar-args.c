// RUN: %clang --target=x86_64-unknown-linux-gnu %s -### --emit-static-lib \
// RUN:     -Xstatic-lib-tool -U -Xstatic-lib-tool --format=gnu 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "{{.*}}llvm-ar" "rcsD" "-U" "--format=gnu" "a.out" "{{.*}}linux-ar-args-{{.*}}.o"

// RUN: %clang --target=x86_64-unknown-linux-gnu %s -### --emit-static-lib \
// RUN:     --static-lib-target-arch-only 2>&1 | FileCheck %s --check-prefix=LIBTOOL-ARG
// LIBTOOL-ARG: warning: argument unused during compilation: '--static-lib-target-arch-only'
// LIBTOOL-ARG: "{{.*}}llvm-ar" "rcsD"
// LIBTOOL-ARG-NOT: "--static-lib-target-arch-only"
// LIBTOOL-ARG-NOT: "-arch_only"
