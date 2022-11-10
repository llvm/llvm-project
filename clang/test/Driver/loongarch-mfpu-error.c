// RUN: %clang --target=loongarch64 -mfpu=xxx -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s

// CHECK: invalid argument 'xxx' to -mfpu=; must be one of: 64, 32, 0, none
