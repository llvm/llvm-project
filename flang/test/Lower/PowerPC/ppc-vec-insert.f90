! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! vec_insert

!CHECK-LABEL: vec_insert_testf32
subroutine vec_insert_testf32(v, x, i1, i2, i4, i8)
  real(4) :: v
  vector(real(4)) :: x
  vector(real(4)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i8 %[[urem]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i16 %[[urem]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i32 %[[urem]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i64 %[[urem]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf32

!CHECK-LABEL: vec_insert_testf64
subroutine vec_insert_testf64(v, x, i1, i2, i4, i8)
  real(8) :: v
  vector(real(8)) :: x
  vector(real(8)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i8 %[[urem]]
! LLVMIR: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i16 %[[urem]]
! LLVMIR: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i32 %[[urem]]
! LLVMIR: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i64 %[[urem]]
! LLVMIR: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf64

!CHECK-LABEL: vec_insert_testi8
subroutine vec_insert_testi8(v, x, i1, i2, i4, i8)
  integer(1) :: v
  vector(integer(1)) :: x
  vector(integer(1)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 16
! LLVMIR: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i8 %[[urem]]
! LLVMIR: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 16
! LLVMIR: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i16 %[[urem]]
! LLVMIR: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 16
! LLVMIR: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i32 %[[urem]]
! LLVMIR: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 16
! LLVMIR: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i64 %[[urem]]
! LLVMIR: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi8

!CHECK-LABEL: vec_insert_testi16
subroutine vec_insert_testi16(v, x, i1, i2, i4, i8)
  integer(2) :: v
  vector(integer(2)) :: x
  vector(integer(2)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 8
! LLVMIR: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i8 %[[urem]]
! LLVMIR: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 8
! LLVMIR: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i16 %[[urem]]
! LLVMIR: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 8
! LLVMIR: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i32 %[[urem]]
! LLVMIR: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 8
! LLVMIR: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i64 %[[urem]]
! LLVMIR: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi16

!CHECK-LABEL: vec_insert_testi32
subroutine vec_insert_testi32(v, x, i1, i2, i4, i8)
  integer(4) :: v
  vector(integer(4)) :: x
  vector(integer(4)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i8 %[[urem]]
! LLVMIR: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i16 %[[urem]]
! LLVMIR: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i32 %[[urem]]
! LLVMIR: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 4
! LLVMIR: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i64 %[[urem]]
! LLVMIR: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi32

!CHECK-LABEL: vec_insert_testi64
subroutine vec_insert_testi64(v, x, i1, i2, i4, i8)
  integer(8) :: v
  vector(integer(8)) :: x
  vector(integer(8)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i8 %[[urem]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i2)

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[urem:.*]] = urem i16 %[[i2]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i16 %[[urem]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[urem:.*]] = urem i32 %[[i4]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i32 %[[urem]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 2
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i64 %[[urem]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi64
