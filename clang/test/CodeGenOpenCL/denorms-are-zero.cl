// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - %s | FileCheck -check-prefix=DENORM-ZERO %s

// Slow FMAF and slow f32 denormals
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu pitcairn %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu pitcairn %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH-OPT %s

// Fast FMAF, but slow f32 denormals
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu tahiti %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu tahiti %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH-OPT %s

// Fast F32 denormals, but slow FMAF
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH-OPT %s

// Fast F32 denormals and fast FMAF
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa -target-cpu gfx900 %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-DENORM %s
// RUN: %clang_cc1 -emit-llvm -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu gfx900 %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH-OPT %s

// RUN: %clang_cc1 -emit-llvm -target-feature +fp32-denormals -target-feature -fp64-fp16-denormals -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu fiji %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FEATURE %s
// RUN: %clang_cc1 -emit-llvm -target-feature +fp32-denormals -target-feature -fp64-fp16-denormals -cl-denorms-are-zero -o - -triple amdgcn--amdhsa -target-cpu pitcairn %s | FileCheck -check-prefixes=AMDGCN,AMDGCN-FEATURE %s



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

// AMDGCN-FLUSH: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="false" {{.*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"
// AMDGCN-FLUSH-OPT: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="true" {{.*}} "target-features"="{{[^"]*}}+fp64-fp16-denormals,{{[^"]*}}-fp32-denormals{{[^"]*}}"

// AMDGCN-DENORM: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="false" {{.*}} "target-features"="{{[^"]*}}+fp32-denormals,{{[^"]*}}+fp64-fp16-denormals{{[^"]*}}"

// AMDGCN-FEATURE: attributes #{{[0-9]*}} = {{{[^}]*}} "denorms-are-zero"="true" {{.*}} "target-features"="{{[^"]*}}+fp32-denormals,{{[^"]*}}-fp64-fp16-denormals{{[^"]*}}"
void f() {}
