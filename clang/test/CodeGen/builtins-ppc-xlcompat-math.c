// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - | FileCheck %s

// CHECK-LABEL: @mtfsb0(
// CHECK:         call void @llvm.ppc.mtfsb0(i32 10)
// CHECK-NEXT:    ret void
//
void mtfsb0 () {
  __mtfsb0 (10);
}

// CHECK-LABEL: @mtfsb1(
// CHECK:         call void @llvm.ppc.mtfsb1(i32 0)
// CHECK-NEXT:    ret void
//
void mtfsb1 () {
  __mtfsb1 (0);
}

// CHECK-LABEL: @mtfsf(
// CHECK:         [[TMP0:%.*]] = uitofp i32 %{{.*}} to double
// CHECK-NEXT:    call void @llvm.ppc.mtfsf(i32 8, double [[TMP0]])
// CHECK-NEXT:    ret void
//
void mtfsf (unsigned int ui) {
  __mtfsf (8, ui);
}

// CHECK-LABEL: @mtfsfi(
// CHECK:         call void @llvm.ppc.mtfsfi(i32 7, i32 15)
// CHECK-NEXT:    ret void
//
void mtfsfi () {
  __mtfsfi (7, 15);
}

// CHECK-LABEL: @fmsub(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store double [[D:%.*]], double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = call double @llvm.ppc.fmsub(double [[TMP0]], double [[TMP1]], double [[TMP2]])
// CHECK-NEXT:    ret double [[TMP3]]
//
double fmsub (double d) {
  return __fmsub (d, d, d);
}

// CHECK-LABEL: @fmsubs(
// CHECK:         [[F_ADDR:%.*]] = alloca float, align 4
// CHECK-NEXT:    store float [[F:%.*]], float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = call float @llvm.ppc.fmsubs(float [[TMP0]], float [[TMP1]], float [[TMP2]])
// CHECK-NEXT:    ret float [[TMP3]]
//
float fmsubs (float f) {
  return __fmsubs (f, f, f);
}

// CHECK-LABEL: @fnmadd(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store double [[D:%.*]], double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = call double @llvm.ppc.fnmadd(double [[TMP0]], double [[TMP1]], double [[TMP2]])
// CHECK-NEXT:    ret double [[TMP3]]
//
double fnmadd (double d) {
  return __fnmadd (d, d, d);
}

// CHECK-LABEL: @fnmadds(
// CHECK:         [[F_ADDR:%.*]] = alloca float, align 4
// CHECK-NEXT:    store float [[F:%.*]], float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = call float @llvm.ppc.fnmadds(float [[TMP0]], float [[TMP1]], float [[TMP2]])
// CHECK-NEXT:    ret float [[TMP3]]
//
float fnmadds (float f) {
  return __fnmadds (f, f, f);
}

// CHECK-LABEL: @fnmsub(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store double [[D:%.*]], double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = call double @llvm.ppc.fnmsub(double [[TMP0]], double [[TMP1]], double [[TMP2]])
// CHECK-NEXT:    ret double [[TMP3]]
//
double fnmsub (double d) {
  return __fnmsub (d, d, d);
}

// CHECK-LABEL: @fnmsubs(
// CHECK:         [[F_ADDR:%.*]] = alloca float, align 4
// CHECK-NEXT:    store float [[F:%.*]], float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = call float @llvm.ppc.fnmsubs(float [[TMP0]], float [[TMP1]], float [[TMP2]])
// CHECK-NEXT:    ret float [[TMP3]]
//
float fnmsubs (float f) {
  return __fnmsubs (f, f, f);
}

// CHECK-LABEL: @fre(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store double [[D:%.*]], double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, double* [[D_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.ppc.fre(double [[TMP0]])
// CHECK-NEXT:    ret double [[TMP1]]
//
double fre (double d) {
  return __fre (d);
}

// CHECK-LABEL: @fres(
// CHECK:         [[F_ADDR:%.*]] = alloca float, align 4
// CHECK-NEXT:    store float [[F:%.*]], float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load float, float* [[F_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = call float @llvm.ppc.fres(float [[TMP0]])
// CHECK-NEXT:    ret float [[TMP1]]
//
float fres (float f) {
  return __fres (f);
}
