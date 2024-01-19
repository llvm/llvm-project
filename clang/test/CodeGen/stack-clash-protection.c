// Check the correct function attributes are generated
// RUN: %clang_cc1 -triple x86_64-linux -O0 -S -emit-llvm -o- %s -fstack-clash-protection -mstack-probe-size=8192 | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -O0 -S -emit-llvm -o- %s -fstack-clash-protection -mstack-probe-size=8192 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -O0 -S -emit-llvm -o- %s -fstack-clash-protection -mstack-probe-size=8192 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -O0 -S -emit-llvm -o- %s -fstack-clash-protection -mstack-probe-size=8192 | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -O0 -S -emit-llvm -o- %s -fstack-clash-protection -mstack-probe-size=8192 | FileCheck %s

// CHECK: define{{.*}} void @large_stack() #[[A:.*]] {
void large_stack(void) {
  volatile int stack[20000], i;
  for (i = 0; i < sizeof(stack) / sizeof(int); ++i)
    stack[i] = i;
}

// CHECK: define{{.*}} void @vla({{.*}}) #[[A]] {
void vla(int n) {
  volatile int vla[n];
  __builtin_memset(&vla[0], 0, 1);
}

// CHECK: define{{.*}} void @builtin_alloca({{.*}}) #[[A]] {
void builtin_alloca(int n) {
  volatile void *mem = __builtin_alloca(n);
}

// CHECK: attributes #[[A]] = {{.*}}"probe-stack"="inline-asm" {{.*}}"stack-probe-size"="8192"

// CHECK: !{i32 4, !"probe-stack", !"inline-asm"}
// CHECK: !{i32 8, !"stack-probe-size", i32 8192}
