// RUN: not %clang -target sparcv9 -march=v9 -### -c %s 2>&1 | FileCheck %s
// RUN: not %clang -target sparc64 -march=v9 -### -c %s 2>&1 | FileCheck %s
// RUN: not %clang -target sparc -march=v9 -### -c %s 2>&1 | FileCheck %s
// CHECK: error: unsupported option '-march=' for target
