// RUN: %clang -target x86_64 %s -Werror -fpic -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// CHECK: "-fsemantic-interposition"

/// No-op for -fno-pic/-fpie.
// RUN: %clang -target x86_64 %s -Werror -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NOOP %s
// RUN: %clang -target x86_64 %s -Werror -fPIE -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NOOP %s
// NOOP-NOT: "-fsemantic-interposition"
// NOOP-NOT: "-fno-semantic-interposition"

/// If -fno-semantic-interposition is specified and the target supports local
/// aliases, neither CC1 option is set.
// RUN: %clang -target aarch64 %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target riscv32 %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target riscv64 %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target i386 %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// NO-NOT: "-fsemantic-interposition"
// NO-NOT: "-fhalf-no-semantic-interposition"

/// If neither -fsemantic-interposition or -fno-semantic-interposition is specified,
/// or -fno-semantic-interposition is specified but the target does not support
/// local aliases, use the traditional half-baked behavor: interprocedural
/// optimizations are allowed but local aliases are not used. If references are
/// not optimized out, semantic interposition at runtime is possible.
// RUN: %clang -target ppc64le %s -Werror -fPIC -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=HALF %s

// RUN: %clang -target x86_64 %s -Werror -fPIC -c -### 2>&1 | FileCheck --check-prefix=HALF %s
//
// HALF: "-fhalf-no-semantic-interposition"
