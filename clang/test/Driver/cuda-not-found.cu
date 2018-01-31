// REQUIRES: clang-driver

// Check that we raise an error if we're trying to compile CUDA code but can't
// find a CUDA install, unless -nocudainc was passed.

// RUN: %clang -### --sysroot=%s/no-cuda-there --cuda-path-ignore-env %s 2>&1 | FileCheck %s --check-prefix ERR
// RUN: %clang -### --cuda-path=%s/no-cuda-there %s 2>&1 | FileCheck %s --check-prefix ERR
// ERR: cannot find CUDA installation

// RUN: %clang -### -nocudainc --sysroot=%s/no-cuda-there --cuda-path-ignore-env %s 2>&1 | FileCheck %s --check-prefix OK
// RUN: %clang -### -nocudainc --cuda-path=%s/no-cuda-there %s 2>&1 | FileCheck %s --check-prefix OK
// OK-NOT: cannot find CUDA installation
