// RUN: %clang_cc1 -verify -x c -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=52  -emit-llvm -o - %s | FileCheck %s
// expected-no-diagnostics

#define N 100
int A[N];

void condlastprivate() {
  int x, y, z, k;
  x = y = z = k = 0;

  #pragma omp parallel for lastprivate(conditional: x,y,z) lastprivate(k)
  for (k = 0; k < N; k++) {
    if ((k > 2) && (k < 6)) {
      x = A[k];
      z = A[k] + 111;
    } else {
      y = A[k] + 222;
    }
  }
}

int main() {
  for (int i = 0; i < N; i++)
    A[i] = i;
  condlastprivate();
  return 0;
}

// CHECK: @.pl_cond.x_[[ID:[0-9]+]].iv = common global i32 0, align 4
// CHECK: @pl_cond.x_[[ID]] = common global i32 0, align 4
// CHECK: @.gomp_critical_user_pl_cond.x_[[ID]].var = common global [8 x i32] zeroinitializer, align 8

// CHECK: @.pl_cond.z_[[ID]].iv = common global i32 0, align 4
// CHECK: @pl_cond.z_[[ID]] = common global i32 0, align 4
// CHECK: @.gomp_critical_user_pl_cond.z_[[ID]].var = common global [8 x i32] zeroinitializer, align 8

// CHECK: @.pl_cond.y_[[ID]].iv = common global i32 0, align 4
// CHECK: @pl_cond.y_[[ID]] = common global i32 0, align 4
// CHECK: @.gomp_critical_user_pl_cond.y_[[ID]].var = common global [8 x i32] zeroinitializer, align 8

// CHECK-LABEL: define internal void @condlastprivate.omp_outlined(
// CHECK: call void @__kmpc_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.x_[[ID]].var)
// CHECK: store i32 %{{[0-9]+}}, ptr @pl_cond.x_[[ID]], align 4
// CHECK: call void @__kmpc_end_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.x_[[ID]].var)

// CHECK: call void @__kmpc_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.z_[[ID]].var)
// CHECK: store i32 %{{[0-9]+}}, ptr @pl_cond.z_[[ID]], align 4
// CHECK: call void @__kmpc_end_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.z_[[ID]].var)

// CHECK: call void @__kmpc_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.y_[[ID]].var)
// CHECK: store i32 %{{[0-9]+}}, ptr @pl_cond.y_[[ID]], align 4
// CHECK: call void @__kmpc_end_critical(ptr @2, {{.*}}, ptr @.gomp_critical_user_pl_cond.y_[[ID]].var)

