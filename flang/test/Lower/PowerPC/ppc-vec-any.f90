! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_any_ge
!----------------------

! CHECK-LABEL: vec_any_ge_test_i1
subroutine vec_any_ge_test_i1(arg1, arg2)
  vector(integer(1)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsb.p(i32 3, <16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_any_ge_test_i1

! CHECK-LABEL: vec_any_ge_test_i2
subroutine vec_any_ge_test_i2(arg1, arg2)
  vector(integer(2)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsh.p(i32 3, <8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_any_ge_test_i2

! CHECK-LABEL: vec_any_ge_test_i4
subroutine vec_any_ge_test_i4(arg1, arg2)
  vector(integer(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsw.p(i32 3, <4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_any_ge_test_i4

! CHECK-LABEL: vec_any_ge_test_i8
subroutine vec_any_ge_test_i8(arg1, arg2)
  vector(integer(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsd.p(i32 3, <2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_any_ge_test_i8

! CHECK-LABEL: vec_any_ge_test_u1
subroutine vec_any_ge_test_u1(arg1, arg2)
  vector(unsigned(1)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtub.p(i32 3, <16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_any_ge_test_u1

! CHECK-LABEL: vec_any_ge_test_u2
subroutine vec_any_ge_test_u2(arg1, arg2)
  vector(unsigned(2)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtuh.p(i32 3, <8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_any_ge_test_u2

! CHECK-LABEL: vec_any_ge_test_u4
subroutine vec_any_ge_test_u4(arg1, arg2)
  vector(unsigned(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtuw.p(i32 3, <4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_any_ge_test_u4

! CHECK-LABEL: vec_any_ge_test_u8
subroutine vec_any_ge_test_u8(arg1, arg2)
  vector(unsigned(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtud.p(i32 3, <2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_any_ge_test_u8

! CHECK-LABEL: vec_any_ge_test_r4
subroutine vec_any_ge_test_r4(arg1, arg2)
  vector(real(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.vsx.xvcmpgesp.p(i32 1, <4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_any_ge_test_r4

! CHECK-LABEL: vec_any_ge_test_r8
subroutine vec_any_ge_test_r8(arg1, arg2)
  vector(real(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call i32 @llvm.ppc.vsx.xvcmpgedp.p(i32 1, <2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_any_ge_test_r8

