! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_abs
!----------------------

! CHECK-LABEL: vec_abs_i1
subroutine vec_abs_i1(arg1)
  vector(integer(1)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[sub:.*]] = sub <16 x i8> zeroinitializer, %[[arg1]]
! LLVMIR: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %[[sub]], <16 x i8> %[[arg1]])
end subroutine vec_abs_i1

! CHECK-LABEL: vec_abs_i2
subroutine vec_abs_i2(arg1)
  vector(integer(2)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[sub:.*]] = sub <8 x i16> zeroinitializer, %[[arg1]]
! LLVMIR: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %[[sub]], <8 x i16> %[[arg1]])
end subroutine vec_abs_i2

! CHECK-LABEL: vec_abs_i4
subroutine vec_abs_i4(arg1)
  vector(integer(4)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[sub:.*]] = sub <4 x i32> zeroinitializer, %[[arg1]]
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %[[sub]], <4 x i32> %[[arg1]])
end subroutine vec_abs_i4

! CHECK-LABEL: vec_abs_i8
subroutine vec_abs_i8(arg1)
  vector(integer(8)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[sub:.*]] = sub <2 x i64> zeroinitializer, %[[arg1]]
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %[[sub]], <2 x i64> %[[arg1]])
end subroutine vec_abs_i8

! CHECK-LABEL: vec_abs_r4
subroutine vec_abs_r4(arg1)
  vector(real(4)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call contract <4 x float> @llvm.fabs.v4f32(<4 x float> %[[arg1]])
end subroutine vec_abs_r4

! CHECK-LABEL: vec_abs_r8
subroutine vec_abs_r8(arg1)
  vector(real(8)) :: arg1, r
  r = vec_abs(arg1)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call contract <2 x double> @llvm.fabs.v2f64(<2 x double> %[[arg1]])
end subroutine vec_abs_r8
