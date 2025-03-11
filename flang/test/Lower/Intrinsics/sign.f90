! RUN: bbc %s -o - | FileCheck %s --check-prefixes=CHECK,%if flang-supports-f128-math %{F128%} %else %{F64%}

! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: %[[VAL_1:.*]] = arith.shrsi %{{.*}}, %{{.*}} : i32
  ! CHECK: %[[VAL_2:.*]] = arith.xori %{{.*}}, %[[VAL_1]] : i32
  ! CHECK: %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[VAL_1]] : i32
  ! CHECK-DAG: %[[VAL_4:.*]] = arith.subi %{{.*}}, %[[VAL_3]] : i32
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
  ! CHECK: select %[[VAL_5]], %[[VAL_4]], %[[VAL_3]] : i32
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-NOT: fir.call @{{.*}}fabs
  ! CHECK: math.copysign{{.*}} : f32
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr2
subroutine sign_testr2(a, b, c)
  integer, parameter :: rk = merge(16, 8, selected_real_kind(33, 4931)==16)
  real(KIND=rk) a, b, c
  ! CHECK-NOT: fir.call @{{.*}}fabs
  ! F128: math.copysign{{.*}} : f128
  ! F64: math.copysign{{.*}} : f64
  c = sign(a, b)
end subroutine
