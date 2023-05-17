/// For most targets -p is legacy. We used to report -Wunused-command-line-argument.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -c -p %s 2>&1 | FileCheck %s --check-prefix=ERR

// RUN: %clang -### --target=x86_64-unknown-openbsd -c -p %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: %clang -### --target=powerpc64-ibm-aix -c -p %s 2>&1 | FileCheck %s --implicit-check-not=error:

// ERR: error: unsupported option '-p' for target {{.*}}
