! RUN: bbc %s -o - | FileCheck %s

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
  ! CHECK: fir.call @{{.*}}copysign{{.*}} : (f32, f32) -> f32
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr2
subroutine sign_testr2(a, b, c)
  real(KIND=16) a, b, c
  ! CHECK-NOT: fir.call @{{.*}}fabs
  ! CHECK: fir.call @{{.*}}copysign{{.*}} : (f128, f128) -> f128
  c = sign(a, b)
end subroutine
