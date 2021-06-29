// Test general-regs-only target attribute on x86

// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

// CHECK: define{{.*}} void @f() [[GPR_ATTRS:#[0-9]+]]
void __attribute__((target("general-regs-only"))) f() { }
// CHECK: define{{.*}} void @f_before() [[GPR_ATTRS:#[0-9]+]]
void __attribute__((target("avx2,general-regs-only"))) f_before() { }
// CHECK: define{{.*}} void @f_after() [[AVX2_ATTRS:#[0-9]+]]
void __attribute__((target("general-regs-only,avx2"))) f_after() { }

// CHECK: attributes [[GPR_ATTRS]] = { {{.*}} "target-features"="{{.*}}-avx{{.*}}-avx2{{.*}}-avx512f{{.*}}-sse{{.*}}-sse2{{.*}}-ssse3{{.*}}-x87{{.*}}"
// CHECK: attributes [[AVX2_ATTRS]] = { {{.*}} "target-features"="{{.*}}+avx{{.*}}+avx2{{.*}}+sse{{.*}}+sse2{{.*}}+ssse3{{.*}}-avx512f{{.*}}-x87{{.*}}"
