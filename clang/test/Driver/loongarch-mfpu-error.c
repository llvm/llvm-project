// RUN: %clang --target=loongarch64 -mfpu=xxx %s -### 2>&1 | FileCheck %s

// CHECK: invalid argument 'xxx' to -mfpu=; must be one of: 64, 32, none, 0 (alias for none)
