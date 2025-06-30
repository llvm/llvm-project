// REQUIRES: loongarch-registered-target
// RUN: %clang_cc1 -emit-llvm -o - -triple loongarch64-linux-gnu %s  \
// RUN:    | FileCheck %s --check-prefix=CHECK
_Float16 ha1 = 0;
_Float16 ha2 = 1;
__float128 qa1 = 0;
__float128 qa2 = 1;
__float128 qresult = 0;
_Float16 hresult;
float fresult;
double dresult;
int status;

void fp16_test() {
  // CHECK-LABEL: @fp16_test
  // CHECK: [[TMP0:%.*]] = load half, ptr @ha1, align 2
  // CHECK-NEXT: store half [[TMP0]], ptr @hresult, align 2
  hresult = ha1;
  // CHECK: [[TMP1:%.*]] = load half, ptr @ha1, align 2
  // CHECK-NEXT: [[UP_RESULT:%.*]] = fpext half [[TMP1]] to float
  // CHECK-NEXT: [[NEG_RESULT:%.*]] = fneg float [[UP_RESULT]]
  // CHECK-NEXT: [[DOWN_RESULTNEG:%.*]] = fptrunc float [[NEG_RESULT]] to half
  // CHECK-NEXT: store half [[DOWN_RESULTNEG]], ptr @hresult, align 2
  hresult = -ha1;
  // CHECK: [[TMP0:%.*]] = load half, ptr @ha1, align 2
  // CHECK-NEXT: [[UP1_RESULT:%.*]] = fpext half [[TMP0]] to float
  // CHECK-NEXT: [[TMP1:%.*]] = load half, ptr @ha2, align 2
  // CHECK-NEXT: [[UP2_RESULT:%.*]] = fpext half [[TMP1]] to float
  // CHECK-NEXT: [[RESULT_ADD:%.*]] = fadd float [[UP1_RESULT]], [[UP2_RESULT]]
  // CHECK-NEXT: [[DOWN_RESULT_ADD:%.*]] = fptrunc float [[RESULT_ADD]] to half
  // CHECK-NEXT: store half [[DOWN_RESULT_ADD]], ptr @hresult, align 2
  hresult = ha1 + ha2;
  // CHECK: [[UP1_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: [[UP2_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: {{%.*}} = fsub float [[UP1_RESULT]], [[UP2_RESULT]]
  hresult = ha1 - ha2;
  // CHECK: [[UP1_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: [[UP2_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: {{%.*}} = fmul float [[UP1_RESULT]], [[UP2_RESULT]]
  hresult = ha1 * ha2;
  // CHECK: [[UP1_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: [[UP2_RESULT:%.*]] = fpext half {{%.*}} to float
  // CHECK: {{%.*}} = fdiv float [[UP1_RESULT]], [[UP2_RESULT]]
  hresult = ha1 / ha2;
  int status = 0;
  // CHECK: [[TMP:%.*]] = fpext half {{%.*}} to float
  // CHECK-NEXT: store float [[TMP]], ptr @fresult, align 4
  fresult = (float)ha1;
  // CHECK: [[TMP:%.*]] = fpext half {{%.*}} to double
  // CHECK-NEXT: store double [[TMP]], ptr @dresult, align 8
  dresult = (double)ha1;
}

void fp128_test() {
  // CHECK-LABEL: @fp128_test
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: store fp128 [[TMPQ0]], ptr {{@.*}}, align 16
  qresult = qa1;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[NEG_RESULT:%.*]] = fneg fp128 [[TMPQ0]]
  // CHECK-NEXT: store fp128 [[NEG_RESULT]], ptr {{@.*}}, align 16
  qresult = -qa1;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[TMPQ1:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[ADD_RESULT:%.*]] = fadd fp128 [[TMPQ0]], [[TMPQ1]]
  // CHECK-NEXT: store fp128 [[ADD_RESULT]], ptr {{@.*}}, align 16
  qresult = qa1 + qa2;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[TMPQ1:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[SUB_RESULT:%.*]] = fsub fp128 [[TMPQ0]], [[TMPQ1]]
  // CHECK-NEXT: store fp128 [[SUB_RESULT]], ptr {{@.*}}, align 16
  qresult = qa1 - qa2;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[TMPQ1:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[MUL_RESULT:%.*]] = fmul fp128 [[TMPQ0]], [[TMPQ1]]
  // CHECK-NEXT: store fp128 [[MUL_RESULT]], ptr {{@.*}}, align 16
  qresult = qa1 * qa2;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[TMPQ1:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[DIV_RESULT:%.*]] = fdiv fp128 [[TMPQ0]], [[TMPQ1]]
  // CHECK-NEXT: store fp128 [[DIV_RESULT]], ptr {{@.*}}, align 16
  qresult = qa1 / qa2;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[RESULT:%.*]] = fptrunc fp128 [[TMPQ0]] to float
  // CHECK-NEXT: store float [[RESULT]], ptr {{@.*}}, align 4

  fresult = (float)qa1;
  // CHECK: [[TMPQ0:%.*]] = load fp128, ptr {{@.*}}, align 16
  // CHECK-NEXT: [[RESULT:%.*]] = fptrunc fp128 [[TMPQ0]] to double
  // CHECK-NEXT: store double [[RESULT]], ptr {{@.*}}, align 8
  dresult = (double)qa1;
}

