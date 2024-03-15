// RUN: %clang_cc1 -triple arm64-apple-xros1 -emit-llvm -o - %s 2>&1 | FileCheck %s

__attribute__((availability(visionOS, introduced=1.1)))
void introduced_1_1();

void use() {
  if (__builtin_available(visionOS 1.2, *))
    introduced_1_1();
  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 11, i32 1, i32 2, i32 0)
}
