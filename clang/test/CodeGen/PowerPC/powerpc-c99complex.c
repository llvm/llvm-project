// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOLDBL128
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOLDBL128
// RUN: %clang_cc1 -triple powerpc64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LDBL128
// RUN: %clang_cc1 -triple ppc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LDBL128
// RUN: %clang_cc1 -triple powerpc-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix=PPC32LNX

_Complex float foo1(_Complex float x) {
  return x;
// CHECK-LABEL:             define{{.*}} { float, float } @foo1(float noundef %x.{{.*}}, float noundef %x.{{.*}}) #0 {
// CHECK:                   ret { float, float }

// PPC32LNX-LABEL:          define{{.*}} void @foo1(ptr dead_on_unwind noalias writable sret({ float, float }) align 4 %agg.result, ptr noundef byval({ float, float }) align 4 %x) #0 {
// PPC32LNX:                [[RETREAL:%.*]] = getelementptr inbounds nuw { float, float }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-NEXT:           [[RETIMAG:%.*]] = getelementptr inbounds nuw { float, float }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-NEXT:           store float %{{.*}}, ptr [[RETREAL]], align 4
// PPC32LNX-NEXT:           store float %{{.*}}, ptr [[RETIMAG]], align 4
}

_Complex double foo2(_Complex double x) {
  return x;
// CHECK-LABEL:             define{{.*}} { double, double } @foo2(double noundef %x.{{.*}}, double noundef %x.{{.*}}) #0 {
// CHECK:                   ret { double, double }

// PPC32LNX-LABEL:          define{{.*}} void @foo2(ptr dead_on_unwind noalias writable sret({ double, double }) align 8 %agg.result, ptr noundef byval({ double, double }) align 8 %x) #0 {
// PPC32LNX:                [[RETREAL:%.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-NEXT:           [[RETIMAG:%.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-NEXT:           store double %{{.*}}, ptr [[RETREAL]], align 8
// PPC32LNX-NEXT:           store double %{{.*}}, ptr [[RETIMAG]], align 8
}

_Complex long double foo3(_Complex long double x) {
  return x;
// CHECK-NOLDBL128-LABEL:   define{{.*}} { double, double } @foo3(double noundef %x.{{.*}}, double noundef %x.{{.*}}) #0 {
// CHECK-NOLDBL128:         ret { double, double }

// CHECK-LDBL128-LABEL:     define{{.*}} { ppc_fp128, ppc_fp128 } @foo3(ppc_fp128 noundef %x.{{.*}}, ppc_fp128 noundef %x.{{.*}}) #0 {
// CHECK-LDBL128:           ret { ppc_fp128, ppc_fp128 }

// PPC32LNX-LABEL:          define{{.*}} void @foo3(ptr dead_on_unwind noalias writable sret({ ppc_fp128, ppc_fp128 }) align 16 %agg.result, ptr noundef byval({ ppc_fp128, ppc_fp128 }) align 16 %x) #0 {
// PPC32LNX:                [[RETREAL:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-NEXT:           [[RETIMAG:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-NEXT:           store ppc_fp128 %{{.*}}, ptr [[RETREAL]], align 16
// PPC32LNX-NEXT:           store ppc_fp128 %{{.*}}, ptr [[RETIMAG]], align 16
}
