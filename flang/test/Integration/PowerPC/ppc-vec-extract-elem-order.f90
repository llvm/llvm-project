!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
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
! LLVMIR: %[[idx:.*]] = zext i8 %[[sub]] to i64
! LLVMIR: %[[r:.*]] = extractelement <2 x i64> %[[arg1]], i64 %[[idx]]
! LLVMIR: store i64 %[[r]], ptr %{{[0-9]}}, align 8
end subroutine vec_extract_testi8i1
