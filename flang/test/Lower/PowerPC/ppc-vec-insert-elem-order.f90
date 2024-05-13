! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!CHECK-LABEL: vec_insert_testf32i64
subroutine vec_insert_testf32i64(v, x, i8)
  real(4) :: v
  vector(real(4)) :: x
  vector(real(4)) :: r
  integer(8) :: i8
  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 4
! LLVMIR: %[[sub:.*]] = sub i64 3, %[[urem]]
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i64 %[[sub]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf32i64

!CHECK-LABEL: vec_insert_testi64i8
subroutine vec_insert_testi64i8(v, x, i1, i2, i4, i8)
  integer(8) :: v
  vector(integer(8)) :: x
  vector(integer(8)) :: r
  integer(1) :: i1
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 2
! LLVMIR: %[[sub:.*]] = sub i8 1, %[[urem]]
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i8 %[[sub]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi64i8
