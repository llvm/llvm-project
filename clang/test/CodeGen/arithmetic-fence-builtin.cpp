// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mreassociate  -o - %s \
// RUN: -opaque-pointers | FileCheck --check-prefix CHECK %s

// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -mreassociate  -o - %s \
// RUN: -opaque-pointers | FileCheck --check-prefix CHECK %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mreassociate \
// RUN: -fprotect-parens -ffp-contract=on -o - %s -opaque-pointers \
// RUN: | FileCheck --check-prefix CHECK %s

template <typename T> T addAF(T a, T b) {
  return __arithmetic_fence(a + b);
}

int addit(float a, float b) {
  // CHECK-LABEL: define {{.*}} @{{.*}}additff(float {{.*}}, float {{.*}}) #0 {
  float af = addAF(a,b);

  // CHECK: [[ADDR_A:%.*]] = alloca float, align 4
  // CHECK-NEXT: [[ADDR_B:%.*]] = alloca float, align 4
  // CHECK-NEXT: [[AF:%.*]] = alloca float, align 4
  // CHECK-NEXT: store float {{.*}}, ptr [[ADDR_A]], align 4
  // CHECK-NEXT: store float {{.*}}, ptr [[ADDR_B]], align 4
  // CHECK-NEXT: [[TEMP_A:%.*]] = load float, ptr [[ADDR_A]], align 4
  // CHECK-NEXT: [[TEMP_B:%.*]] = load float, ptr [[ADDR_B]], align 4
  // CHECK-NEXT: [[CALL2:%.*]] = call reassoc noundef float @_Z5addAFIfET_S0_S0_(float noundef [[TEMP_A]], float noundef [[TEMP_B]])
  // CHECK-NEXT:  store float [[CALL2]], ptr [[AF]], align 4

  return 0;
  // CHECK-NEXT ret i32 0
}

  // CHECK-LABEL: define linkonce_odr noundef float @_Z5addAFIfET_S0_S0_(float noundef {{.*}}, float noundef {{.*}})
  // CHECK: [[A:%.*]] = alloca float, align 4
  // CHECK-NEXT: [[B:%.*]] = alloca float, align 4
  // CHECK-NEXT: store float {{.*}}, ptr [[A]], align 4
  // CHECK-NEXT: store float {{.*}}, ptr [[B]], align 4
  // CHECK-NEXT: [[Z1:%.*]] = load float, ptr [[A]], align 4
  // CHECK-NEXT: [[Z2:%.*]] = load float, ptr [[B]], align 4
  // CHECK-NEXT: [[ADD:%.*]] = fadd reassoc float [[Z1]], [[Z2]]
  // CHECK-NEXT: [[RES:%.*]] = call reassoc float @llvm.arithmetic.fence.f32(float [[ADD]])
  // CHECK: ret float [[RES]]
