! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_cmpge
!----------------------

! CHECK-LABEL: vec_cmpge_test_i8
subroutine vec_cmpge_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmpge_test_i8

! CHECK-LABEL: vec_cmpge_test_i4
subroutine vec_cmpge_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmpge_test_i4

! CHECK-LABEL: vec_cmpge_test_i2
subroutine vec_cmpge_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmpge_test_i2

! CHECK-LABEL: vec_cmpge_test_i1
subroutine vec_cmpge_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmpge_test_i1

! CHECK-LABEL: vec_cmpge_test_u8
subroutine vec_cmpge_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmpge_test_u8

! CHECK-LABEL: vec_cmpge_test_u4
subroutine vec_cmpge_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmpge_test_u4

! CHECK-LABEL: vec_cmpge_test_u2
subroutine vec_cmpge_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmpge_test_u2

! CHECK-LABEL: vec_cmpge_test_u1
subroutine vec_cmpge_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmpge_test_u1

subroutine vec_cmpge_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_cmpge_test_r4

subroutine vec_cmpge_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_cmpge_test_r8

!----------------------
! vec_cmpgt
!----------------------

! CHECK-LABEL: vec_cmpgt_test_i1
subroutine vec_cmpgt_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
end subroutine vec_cmpgt_test_i1

! CHECK-LABEL: vec_cmpgt_test_i2
subroutine vec_cmpgt_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
end subroutine vec_cmpgt_test_i2

! CHECK-LABEL: vec_cmpgt_test_i4
subroutine vec_cmpgt_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_cmpgt_test_i4

! CHECK-LABEL: vec_cmpgt_test_i8
subroutine vec_cmpgt_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
end subroutine vec_cmpgt_test_i8

! CHECK-LABEL: vec_cmpgt_test_u1
subroutine vec_cmpgt_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
end subroutine vec_cmpgt_test_u1

! CHECK-LABEL: vec_cmpgt_test_u2
subroutine vec_cmpgt_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
end subroutine vec_cmpgt_test_u2

! CHECK-LABEL: vec_cmpgt_test_u4
subroutine vec_cmpgt_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_cmpgt_test_u4

! CHECK-LABEL: vec_cmpgt_test_u8
subroutine vec_cmpgt_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
end subroutine vec_cmpgt_test_u8

! CHECK-LABEL: vec_cmpgt_test_r4
subroutine vec_cmpgt_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_cmpgt_test_r4

! CHECK-LABEL: vec_cmpgt_test_r8
subroutine vec_cmpgt_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_cmpgt_test_r8

!----------------------
! vec_cmple
!----------------------

! CHECK-LABEL: vec_cmple_test_i8
subroutine vec_cmple_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmple_test_i8

! CHECK-LABEL: vec_cmple_test_i4
subroutine vec_cmple_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmple_test_i4

! CHECK-LABEL: vec_cmple_test_i2
subroutine vec_cmple_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmple_test_i2

! CHECK-LABEL: vec_cmple_test_i1
subroutine vec_cmple_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmple_test_i1

! CHECK-LABEL: vec_cmple_test_u8
subroutine vec_cmple_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmple_test_u8

! CHECK-LABEL: vec_cmple_test_u4
subroutine vec_cmple_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmple_test_u4

! CHECK-LABEL: vec_cmple_test_u2
subroutine vec_cmple_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmple_test_u2

! CHECK-LABEL: vec_cmple_test_u1
subroutine vec_cmple_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmple_test_u1

! CHECK-LABEL: vec_cmple_test_r4
subroutine vec_cmple_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %[[arg2]], <4 x float> %[[arg1]])
end subroutine vec_cmple_test_r4

! CHECK-LABEL: vec_cmple_test_r8
subroutine vec_cmple_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %[[arg2]], <2 x double> %[[arg1]])
end subroutine vec_cmple_test_r8

!----------------------
! vec_cmplt
!----------------------

! CHECK-LABEL: vec_cmplt_test_i1
subroutine vec_cmplt_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_cmplt_test_i1

! CHECK-LABEL: vec_cmplt_test_i2
subroutine vec_cmplt_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_cmplt_test_i2

! CHECK-LABEL: vec_cmplt_test_i4
subroutine vec_cmplt_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_cmplt_test_i4

! CHECK-LABEL: vec_cmplt_test_i8
subroutine vec_cmplt_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_cmplt_test_i8

! CHECK-LABEL: vec_cmplt_test_u1
subroutine vec_cmplt_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_cmplt_test_u1

! CHECK-LABEL: vec_cmplt_test_u2
subroutine vec_cmplt_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_cmplt_test_u2

! CHECK-LABEL: vec_cmplt_test_u4
subroutine vec_cmplt_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_cmplt_test_u4

! CHECK-LABEL: vec_cmplt_test_u8
subroutine vec_cmplt_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_cmplt_test_u8

! CHECK-LABEL: vec_cmplt_test_r4
subroutine vec_cmplt_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %[[arg2]], <4 x float> %[[arg1]])
end subroutine vec_cmplt_test_r4

! CHECK-LABEL: vec_cmplt_test_r8
subroutine vec_cmplt_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %[[arg2]], <2 x double> %[[arg1]])
end subroutine vec_cmplt_test_r8

