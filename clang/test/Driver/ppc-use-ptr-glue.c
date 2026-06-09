// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -maix-use-ptrgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=PTR_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -maix-use-ptrgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=PTR_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mno-aix-use-ptrgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -maix-use-ptrgl -mno-aix-use-ptrgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=INLINE_GLUE

// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -maix-use-ptrgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR

// PTR_GLUE: "-target-feature" "+use-ptrgl-helper"
// INLINE_GLUE-NOT: "+use-ptrgl-helper"
// ERR: error: unsupported option '-maix-use-ptrgl' for target 'powerpc64le-unknown-linux-gnu'
