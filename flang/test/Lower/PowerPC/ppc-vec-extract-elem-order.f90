! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!CHECK-LABEL: vec_extract_testr4i8
subroutine vec_extract_testr4i8(arg1, arg2, r)
  vector(real(4)) :: arg1
  real(4) :: r
  integer(8) :: arg2
  r = vec_extract(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[arg2]], 4
! LLVMIR: %[[sub:.*]] = sub i64 3, %[[urem]]
! LLVMIR: %[[r:.*]] = extractelement <4 x float> %[[arg1]], i64 %[[sub]]
! LLVMIR: store float %[[r]], ptr %{{[0-9]}}, align 4
end subroutine vec_extract_testr4i8

!CHECK-LABEL: vec_extract_testi8i1
subroutine vec_extract_testi8i1(arg1, arg2, r)
  vector(integer(8)) :: arg1
  integer(8) :: r
  integer(1) :: arg2
  r = vec_extract(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[arg2:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[arg2]], 2
! LLVMIR: %[[sub:.*]] = sub i8 1, %[[urem]]
! LLVMIR: %[[r:.*]] = extractelement <2 x i64> %[[arg1]], i8 %[[sub]]
! LLVMIR: store i64 %[[r]], ptr %{{[0-9]}}, align 8
end subroutine vec_extract_testi8i1
