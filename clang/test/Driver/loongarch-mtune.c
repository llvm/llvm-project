// RUN: %clang --target=loongarch64 -mtune=loongarch64 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=loongarch64
// RUN: %clang --target=loongarch64 -mtune=loongarch64 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRATTR -DCPU=loongarch64

// RUN: %clang --target=loongarch64 -mtune=la464 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=la464
// RUN: %clang --target=loongarch64 -mtune=la464 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRATTR -DCPU=la464

// RUN: %clang --target=loongarch64 -mtune=la664 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=la664
// RUN: %clang --target=loongarch64 -mtune=la664 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRATTR -DCPU=la664

// RUN: %clang --target=loongarch64 -mtune=invalidcpu -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=invalidcpu
// RUN: not %clang --target=loongarch64 -mtune=invalidcpu -S -emit-llvm %s -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR -DCPU=invalidcpu

// RUN: %clang --target=loongarch64 -mtune=generic -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=generic
// RUN: not %clang --target=loongarch64 -mtune=generic -S -emit-llvm %s -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR -DCPU=generic

// RUN: %clang --target=loongarch64 -mtune=generic-la64 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=generic-la64
// RUN: not %clang --target=loongarch64 -mtune=generic-la64 -S -emit-llvm %s -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR -DCPU=generic-la64

// CC1ARG: "-tune-cpu" "[[CPU]]"
// IRATTR: "tune-cpu"="[[CPU]]"

// ERROR: error: unknown target CPU '[[CPU]]'
// ERROR-NEXT: note: valid target CPU values are: {{.*}}

int foo(void) {
  return 3;
}
