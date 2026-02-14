// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

// CHECK: gh_180563
int gh_180563(int __attribute__((vector_size(8))) v) {
  return v[~0UL];
}
