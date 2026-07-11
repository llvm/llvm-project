// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-NOLDBL128,CHECK
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-NOLDBL128,CHECK
// RUN: %clang_cc1 -triple powerpc64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-LDBL128,CHECK
// RUN: %clang_cc1 -triple ppc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-LDBL128,CHECK
// RUN: %clang_cc1 -triple powerpc-unknown-linux -fclang-abi-compat=22 -emit-llvm %s -o - | FileCheck %s --check-prefix=PPC32LNX-CLANG22
// RUN: %clang_cc1 -triple powerpc-unknown-linux -fclang-abi-compat=23 -emit-llvm %s -o - | FileCheck %s --check-prefix=PPC32LNX-CLANG23

_Complex float foo1(_Complex float x) {
  return x;
// CHECK-LABEL:             define{{.*}} { float, float } @foo1(float noundef %x.{{.*}}, float noundef %x.{{.*}}) #0 {
// CHECK:                   ret { float, float }

// PPC32LNX-CLANG22-LABEL:  define{{.*}} void @foo1(ptr dead_on_unwind noalias writable sret({ float, float }) align 4 %agg.result, ptr noundef byval({ float, float }) align 4 %x) #0 {
// PPC32LNX-CLANG22:        [[RETREAL:%.*]] = getelementptr inbounds nuw { float, float }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-CLANG22-NEXT:   [[RETIMAG:%.*]] = getelementptr inbounds nuw { float, float }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-CLANG22-NEXT:   store float %{{.*}}, ptr [[RETREAL]], align 4
// PPC32LNX-CLANG22-NEXT:   store float %{{.*}}, ptr [[RETIMAG]], align 4

// PPC32LNX-CLANG23-LABEL: define dso_local <2 x i32> @foo1(
// PPC32LNX-CLANG23-SAME: <2 x i32> noundef [[X_COERCE:%.*]]) #[[ATTR0:[0-9]+]] {
// PPC32LNX-CLANG23-NEXT:  [[ENTRY:.*:]]
// PPC32LNX-CLANG23-NEXT:    [[RETVAL:%.*]] = alloca { float, float }, align 4
// PPC32LNX-CLANG23-NEXT:    [[X:%.*]] = alloca { float, float }, align 4
// PPC32LNX-CLANG23-NEXT:    store <2 x i32> [[X_COERCE]], ptr [[X]], align 4
// PPC32LNX-CLANG23-NEXT:    [[X_REALP:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[X]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[X_REAL:%.*]] = load float, ptr [[X_REALP]], align 4
// PPC32LNX-CLANG23-NEXT:    [[X_IMAGP:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[X]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    [[X_IMAG:%.*]] = load float, ptr [[X_IMAGP]], align 4
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_REALP:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[RETVAL]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_IMAGP:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[RETVAL]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    store float [[X_REAL]], ptr [[RETVAL_REALP]], align 4
// PPC32LNX-CLANG23-NEXT:    store float [[X_IMAG]], ptr [[RETVAL_IMAGP]], align 4
// PPC32LNX-CLANG23-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[RETVAL]], align 4
// PPC32LNX-CLANG23-NEXT:    ret <2 x i32> [[TMP0]]
}

_Complex double foo2(_Complex double x) {
  return x;
// CHECK-LABEL:             define{{.*}} { double, double } @foo2(double noundef %x.{{.*}}, double noundef %x.{{.*}}) #0 {
// CHECK:                   ret { double, double }

// PPC32LNX-CLANG22-LABEL:  define{{.*}} void @foo2(ptr dead_on_unwind noalias writable sret({ double, double }) align 8 %agg.result, ptr noundef byval({ double, double }) align 8 %x) #0 {
// PPC32LNX-CLANG22:        [[RETREAL:%.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-CLANG22-NEXT:   [[RETIMAG:%.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-CLANG22-NEXT:   store double %{{.*}}, ptr [[RETREAL]], align 8
// PPC32LNX-CLANG22-NEXT:   store double %{{.*}}, ptr [[RETIMAG]], align 8

// PPC32LNX-CLANG23-LABEL: define dso_local [4 x i32] @foo2(
// PPC32LNX-CLANG23-SAME: [4 x i32] noundef [[X_COERCE:%.*]]) #[[ATTR0]] {
// PPC32LNX-CLANG23-NEXT:  [[ENTRY:.*:]]
// PPC32LNX-CLANG23-NEXT:    [[RETVAL:%.*]] = alloca { double, double }, align 8
// PPC32LNX-CLANG23-NEXT:    [[X:%.*]] = alloca { double, double }, align 8
// PPC32LNX-CLANG23-NEXT:    store [4 x i32] [[X_COERCE]], ptr [[X]], align 8
// PPC32LNX-CLANG23-NEXT:    [[X_REALP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[X]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[X_REAL:%.*]] = load double, ptr [[X_REALP]], align 8
// PPC32LNX-CLANG23-NEXT:    [[X_IMAGP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[X]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    [[X_IMAG:%.*]] = load double, ptr [[X_IMAGP]], align 8
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_REALP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[RETVAL]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_IMAGP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[RETVAL]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    store double [[X_REAL]], ptr [[RETVAL_REALP]], align 8
// PPC32LNX-CLANG23-NEXT:    store double [[X_IMAG]], ptr [[RETVAL_IMAGP]], align 8
// PPC32LNX-CLANG23-NEXT:    [[TMP0:%.*]] = load [4 x i32], ptr [[RETVAL]], align 8
// PPC32LNX-CLANG23-NEXT:    ret [4 x i32] [[TMP0]]
}

_Complex long double foo3(_Complex long double x) {
  return x;
// CHECK-NOLDBL128-LABEL:   define{{.*}} { double, double } @foo3(double noundef %x.{{.*}}, double noundef %x.{{.*}}) #0 {
// CHECK-NOLDBL128:         ret { double, double }

// CHECK-LDBL128-LABEL:     define{{.*}} { ppc_fp128, ppc_fp128 } @foo3(ppc_fp128 noundef %x.{{.*}}, ppc_fp128 noundef %x.{{.*}}) #0 {
// CHECK-LDBL128:           ret { ppc_fp128, ppc_fp128 }

// PPC32LNX-CLANG22-LABEL:  define{{.*}} void @foo3(ptr dead_on_unwind noalias writable sret({ ppc_fp128, ppc_fp128 }) align 16 %agg.result, ptr noundef byval({ ppc_fp128, ppc_fp128 }) align 16 %x) #0 {
// PPC32LNX-CLANG22:        [[RETREAL:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr %agg.result, i32 0, i32 0
// PPC32LNX-CLANG22-NEXT:   [[RETIMAG:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr %agg.result, i32 0, i32 1
// PPC32LNX-CLANG22-NEXT:   store ppc_fp128 %{{.*}}, ptr [[RETREAL]], align 16
// PPC32LNX-CLANG22-NEXT:   store ppc_fp128 %{{.*}}, ptr [[RETIMAG]], align 16

// PPC32LNX-CLANG23-LABEL: define dso_local [2 x i128] @foo3(
// PPC32LNX-CLANG23-SAME: [2 x i128] noundef [[X_COERCE:%.*]]) #[[ATTR0]] {
// PPC32LNX-CLANG23-NEXT:  [[ENTRY:.*:]]
// PPC32LNX-CLANG23-NEXT:    [[RETVAL:%.*]] = alloca { ppc_fp128, ppc_fp128 }, align 16
// PPC32LNX-CLANG23-NEXT:    [[X:%.*]] = alloca { ppc_fp128, ppc_fp128 }, align 16
// PPC32LNX-CLANG23-NEXT:    store [2 x i128] [[X_COERCE]], ptr [[X]], align 16
// PPC32LNX-CLANG23-NEXT:    [[X_REALP:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr [[X]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[X_REAL:%.*]] = load ppc_fp128, ptr [[X_REALP]], align 16
// PPC32LNX-CLANG23-NEXT:    [[X_IMAGP:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr [[X]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    [[X_IMAG:%.*]] = load ppc_fp128, ptr [[X_IMAGP]], align 16
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_REALP:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr [[RETVAL]], i32 0, i32 0
// PPC32LNX-CLANG23-NEXT:    [[RETVAL_IMAGP:%.*]] = getelementptr inbounds nuw { ppc_fp128, ppc_fp128 }, ptr [[RETVAL]], i32 0, i32 1
// PPC32LNX-CLANG23-NEXT:    store ppc_fp128 [[X_REAL]], ptr [[RETVAL_REALP]], align 16
// PPC32LNX-CLANG23-NEXT:    store ppc_fp128 [[X_IMAG]], ptr [[RETVAL_IMAGP]], align 16
// PPC32LNX-CLANG23-NEXT:    [[TMP0:%.*]] = load [2 x i128], ptr [[RETVAL]], align 16
// PPC32LNX-CLANG23-NEXT:    ret [2 x i128] [[TMP0]]
}
