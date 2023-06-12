/// AIX-specific link options are rejected for other targets.
// RUN: %clang -### --target=powerpc64-unknown-linux-gnu \
// RUN:   --sysroot %S/Inputs/aix_ppc_tree --unwindlib=libunwind --rtlib=compiler-rt \
// RUN:   -b one -K -mxcoff-build-id=a %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: %clang -### --target=powerpc64-unknown-linux-gnu -c \
// RUN:   --sysroot %S/Inputs/aix_ppc_tree --unwindlib=libunwind --rtlib=compiler-rt \
// RUN:   -b one -K -mxcoff-build-id=a %s 2>&1 | FileCheck %s --implicit-check-not=error:
// CHECK: error: unsupported option '-b' for target '{{.*}}'
// CHECK: error: unsupported option '-K' for target '{{.*}}'
// CHECK: error: unsupported option '-mxcoff-build-id=' for target '{{.*}}'
