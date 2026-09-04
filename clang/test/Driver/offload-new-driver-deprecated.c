// RUN: %clang -### --target=x86_64-linux-gnu --no-offload-new-driver -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN %s
// WARN: warning: argument '--no-offload-new-driver' is deprecated, the legacy offloading driver has been removed
// WARN-NOT: warning: argument '--no-offload-new-driver' is deprecated

// RUN: %clang -### --target=x86_64-linux-gnu --no-offload-new-driver \
// RUN:   --no-offload-new-driver -c %s 2>&1 | FileCheck --check-prefix=WARN %s

// RUN: %clang -### --target=x86_64-linux-gnu -fno-openmp-new-driver -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=ALIAS %s
// ALIAS: warning: argument '-fno-openmp-new-driver' is deprecated, the legacy offloading driver has been removed

// RUN: %clang -### --target=x86_64-linux-gnu --offload-new-driver -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=QUIET %s
// QUIET-NOT: warning:
