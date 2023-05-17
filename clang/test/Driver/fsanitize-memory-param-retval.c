// RUN: %clang -target i386-gnu-linux %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target aarch64-linux-gnu %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-linux-gnu %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-linux-gnu %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=kernel-memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck %s

// CHECK: "-fno-sanitize-memory-param-retval"

// RUN: %clang -target aarch64-linux-gnu -fsyntax-only %s -fsanitize=memory -fno-sanitize-memory-param-retval -c -### 2>&1 | FileCheck --check-prefix=11 %s
// 11: "-fno-sanitize-memory-param-retval"

// RUN: not %clang -target x86_64-linux-gnu -fsyntax-only %s -fsanitize=memory -fno-sanitize-memory-param-retval=1 2>&1 | FileCheck --check-prefix=EXCESS %s
// EXCESS: error: unknown argument: '-fno-sanitize-memory-param-retval=
