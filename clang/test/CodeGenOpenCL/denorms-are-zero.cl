// RUN: %clang_cc1 -S -cl-denorms-are-zero -o - %s 2>&1
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck %s --check-prefix=CHECK-DENORM
// RUN: %clang_cc1 -emit-llvm -target-feature +fp32-denormals -target-feature -fp64-fp16-denormals -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck --check-prefix=CHECK-FEATURE %s

// For non-amdgcn targets, this test just checks that the -cl-denorms-are-zero argument is accepted
// by clang.  This option is currently a no-op, which is allowed by the
// OpenCL specification.

// For amdgcn target cpu fiji, fp32 should be flushed since fiji does not support fp32 denormals, unless +fp32-denormals is
// explicitly set. amdgcn target always do not flush fp64 denormals. The control for fp64 and fp16 denormals is the same.

// CHECK-DENORM-LABEL: define void @f()
// CHECK-DENORM: attributes #{{[0-9]*}} = {{{[^}]*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"
// CHECK-LABEL: define void @f()
// CHECK: attributes #{{[0-9]*}} = {{{[^}]*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"
// CHECK-FEATURE-LABEL: define void @f()
// CHECK-FEATURE: attributes #{{[0-9]*}} = {{{[^}]*}} "target-features"="{{[^"]*}}+fp32-denormals,{{[^"]*}}-fp64-fp16-denormals{{[^"]*}}"
void f() {}
