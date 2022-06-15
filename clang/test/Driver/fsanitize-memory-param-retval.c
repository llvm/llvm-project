// RUN: %clang -target i386-gnu-linux %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target aarch64-linux-gnu %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-linux-gnu %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-linux-gnu %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=kernel-memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck %s

// CHECK: "-fsanitize-memory-param-retval"

// RUN: %clang -target aarch64-linux-gnu -fsyntax-only %s -fsanitize=memory -fsanitize-memory-param-retval -c -### 2>&1 | FileCheck --check-prefix=11 %s
// 11: "-fsanitize-memory-param-retval"

// RUN: not %clang -target x86_64-linux-gnu -fsyntax-only %s -fsanitize=memory -fsanitize-memory-param-retval=1 2>&1 | FileCheck --check-prefix=EXCESS %s
// EXCESS: error: unknown argument: '-fsanitize-memory-param-retval=
