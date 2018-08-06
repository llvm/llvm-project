// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - %s | FileCheck %s --check-prefix=DENORM-ZERO
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck %s --check-prefix=AMDGCN-DENORM
// RUN: %clang_cc1 -emit-llvm -target-feature +fp32-denormals -target-feature -fp64-fp16-denormals -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck --check-prefix=AMDGCN-FEATURE %s

// For all targets 'denorms-are-zero' attribute is set to 'true'
// if '-cl-denorms-are-zero' was specified and  to 'false' otherwise.

// CHECK-LABEL: define {{(dso_local )?}}void @f()
// CHECK: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="false"
//
// DENORM-ZERO-LABEL: define {{(dso_local )?}}void @f()
// DENORM-ZERO: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="true"

// For amdgcn target cpu fiji, fp32 should be flushed since fiji does not support fp32 denormals, unless +fp32-denormals is
// explicitly set. amdgcn target always do not flush fp64 denormals. The control for fp64 and fp16 denormals is the same.

// AMDGCN-LABEL: define void @f()
// AMDGCN: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="true" {{.*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"
// AMDGCN-DENORM-LABEL: define void @f()
// AMDGCN-DENORM: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="false" {{.*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"
// AMDGCN-FEATURE-LABEL: define void @f()
// AMDGCN-FEATURE: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="true" {{.*}} "target-features"="{{[^"]*}}+fp32-denormals,{{[^"]*}}-fp64-fp16-denormals{{[^"]*}}"
void f() {}
