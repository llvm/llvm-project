// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -S -emit-llvm -fsanitize=alignment %s -o - | FileCheck %s

struct alignas(32) S1 {
  int x;
  S1();
};

struct alignas(32) S2 {
  int x;
};

struct alignas(32) S3 {
  int x;
  S3(int *p = new int[4]);
};

struct S4 : public S3 {
  S4() : S3() {}
};

typedef __attribute__((ext_vector_type(2), aligned(32))) float float32x2_t;

struct S5 {
  float32x2_t x;
};

void *operator new (unsigned long, void *p) { return p; }
void *operator new[] (unsigned long, void *p) { return p; }

S1 *func_01() {
  // CHECK-LABEL: define {{.*}} @_Z7func_01v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       call void @_ZN2S1C1Ev(
  // CHECK-NOT:   and i64 %{{.*}}, 31
  // CHECK:       ret %struct.S1*
  return new S1[20];
}

S2 *func_02() {
  // CHECK-LABEL: define {{.*}} @_Z7func_02v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret %struct.S2*
  return new S2;
}

S2 *func_03() {
  // CHECK-LABEL: define {{.*}} @_Z7func_03v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64 %{{.*}}, 31
  // CHECK:       ret %struct.S2*
  return new S2[20];
}

float32x2_t *func_04() {
  // CHECK-LABEL: define {{.*}} @_Z7func_04v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret <2 x float>*
  return new float32x2_t;
}

float32x2_t *func_05() {
  // CHECK-LABEL: define {{.*}} @_Z7func_05v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64 %{{.*}}, 31
  // CHECK:       ret <2 x float>*
  return new float32x2_t[20];
}

S3 *func_07() {
  // CHECK-LABEL: define {{.*}} @_Z7func_07v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       and i64 %{{.*}}, 3, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret %struct.S3*
  return new S3;
}

S3 *func_08() {
  // CHECK-LABEL: define {{.*}} @_Z7func_08v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       and i64 %{{.*}}, 3, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret %struct.S3*
  return new S3[10];
}


S2 *func_10(void *p) {
  // CHECK-LABEL: define {{.*}} @_Z7func_10Pv
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret %struct.S2*
  return new(p) S2;
}

S2 *func_11(void *p) {
  // CHECK-LABEL: define {{.*}} @_Z7func_11Pv
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64 %{{.*}}, 31, !nosanitize
  // CHECK-NOT:   icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret %struct.S2*
  return new(p) S2[10];
}

float32x2_t *func_12() {
  // CHECK-LABEL: define {{.*}} @_Z7func_12v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK:       ret <2 x float>*
  return new float32x2_t;
}

float32x2_t *func_13() {
  // CHECK-LABEL: define {{.*}} @_Z7func_13v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64 %{{.*}}, 31
  // CHECK:       ret <2 x float>*
  return new float32x2_t[20];
}

S4 *func_14() {
  // CHECK-LABEL: define {{.*}} @_Z7func_14v
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64 %{{.*}}, 31
  // CHECK:       ret %struct.S4*
  return new S4;
}

S5 *func_15(const S5 *ptr) {
  // CHECK-LABEL: define {{.*}} @_Z7func_15PK2S5
  // CHECK:       and i64 %{{.*}}, 31, !nosanitize
  // CHECK:       icmp eq i64 %{{.*}}, 0, !nosanitize
  // CHECK-NOT:   and i64
  // CHECK:       ret %struct.S5*
  return new S5(*ptr);
}
