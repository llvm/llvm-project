! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_sl
!----------------------

! CHECK-LABEL: vec_sl_i1
subroutine vec_sl_i1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! LLVMIR: %7 = shl <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sl_i1

! CHECK-LABEL: vec_sl_i2
subroutine vec_sl_i2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! LLVMIR: %7 = shl <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sl_i2

! CHECK-LABEL: vec_sl_i4
subroutine vec_sl_i4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! LLVMIR: %7 = shl <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sl_i4

! CHECK-LABEL: vec_sl_i8
subroutine vec_sl_i8(arg1, arg2)
  vector(integer(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! LLVMIR: %7 = shl <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sl_i8

! CHECK-LABEL: vec_sl_u1
subroutine vec_sl_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! LLVMIR: %7 = shl <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sl_u1

! CHECK-LABEL: vec_sl_u2
subroutine vec_sl_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! LLVMIR: %7 = shl <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sl_u2

! CHECK-LABEL: vec_sl_u4
subroutine vec_sl_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! LLVMIR: %7 = shl <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sl_u4

! CHECK-LABEL: vec_sl_u8
subroutine vec_sl_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! LLVMIR: %{{[0-9]+}} = shl <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sl_u8

!----------------------
! vec_sll
!----------------------
! CHECK-LABEL: vec_sll_i1u1
subroutine vec_sll_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u1

! CHECK-LABEL: vec_sll_i2u1
subroutine vec_sll_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u1

! CHECK-LABEL: vec_sll_i4u1
subroutine vec_sll_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_i4u1

! CHECK-LABEL: vec_sll_i1u2
subroutine vec_sll_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u2

! CHECK-LABEL: vec_sll_i2u2
subroutine vec_sll_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u2

! CHECK-LABEL: vec_sll_i4u2
subroutine vec_sll_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_i4u2

! CHECK-LABEL: vec_sll_i1u4
subroutine vec_sll_i1u4(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u4

! CHECK-LABEL: vec_sll_i2u4
subroutine vec_sll_i2u4(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u4

! CHECK-LABEL: vec_sll_i4u4
subroutine vec_sll_i4u4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_sll_i4u4

! CHECK-LABEL: vec_sll_u1u1
subroutine vec_sll_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u1

! CHECK-LABEL: vec_sll_u2u1
subroutine vec_sll_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u1

! CHECK-LABEL: vec_sll_u4u1
subroutine vec_sll_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_u4u1

! CHECK-LABEL: vec_sll_u1u2
subroutine vec_sll_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u2

! CHECK-LABEL: vec_sll_u2u2
subroutine vec_sll_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u2

! CHECK-LABEL: vec_sll_u4u2
subroutine vec_sll_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_u4u2

! CHECK-LABEL: vec_sll_u1u4
subroutine vec_sll_u1u4(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u4

! CHECK-LABEL: vec_sll_u2u4
subroutine vec_sll_u2u4(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u4

! CHECK-LABEL: vec_sll_u4u4
subroutine vec_sll_u4u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_sll_u4u4

!----------------------
! vec_slo
!----------------------

! CHECK-LABEL: vec_slo_i1u1
subroutine vec_slo_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_i1u1

! CHECK-LABEL: vec_slo_i2u1
subroutine vec_slo_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_i2u1

! CHECK-LABEL: vec_slo_i4u1
subroutine vec_slo_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_i4u1

! CHECK-LABEL: vec_slo_u1u1
subroutine vec_slo_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_u1u1

! CHECK-LABEL: vec_slo_u2u1
subroutine vec_slo_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_u2u1

! CHECK-LABEL: vec_slo_u4u1
subroutine vec_slo_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_u4u1

! CHECK-LABEL: vec_slo_r4u1
subroutine vec_slo_r4u1(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_slo_r4u1

! CHECK-LABEL: vec_slo_i1u2
subroutine vec_slo_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_i1u2

! CHECK-LABEL: vec_slo_i2u2
subroutine vec_slo_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_i2u2

! CHECK-LABEL: vec_slo_i4u2
subroutine vec_slo_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_i4u2

! CHECK-LABEL: vec_slo_u1u2
subroutine vec_slo_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_u1u2

! CHECK-LABEL: vec_slo_u2u2
subroutine vec_slo_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>

end subroutine vec_slo_u2u2

! CHECK-LABEL: vec_slo_u4u2
subroutine vec_slo_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_u4u2

! CHECK-LABEL: vec_slo_r4u2
subroutine vec_slo_r4u2(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_slo_r4u2

!----------------------
! vec_sr
!----------------------
! CHECK-LABEL: vec_sr_i1
subroutine vec_sr_i1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! LLVMIR: %7 = lshr <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sr_i1

! CHECK-LABEL: vec_sr_i2
subroutine vec_sr_i2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! LLVMIR: %7 = lshr <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sr_i2

! CHECK-LABEL: vec_sr_i4
subroutine vec_sr_i4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! LLVMIR: %7 = lshr <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sr_i4

! CHECK-LABEL: vec_sr_i8
subroutine vec_sr_i8(arg1, arg2)
  vector(integer(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! LLVMIR: %7 = lshr <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sr_i8

! CHECK-LABEL: vec_sr_u1
subroutine vec_sr_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! LLVMIR: %7 = lshr <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sr_u1

! CHECK-LABEL: vec_sr_u2
subroutine vec_sr_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! LLVMIR: %7 = lshr <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sr_u2

! CHECK-LABEL: vec_sr_u4
subroutine vec_sr_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! LLVMIR: %7 = lshr <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sr_u4

! CHECK-LABEL: vec_sr_u8
subroutine vec_sr_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! LLVMIR: %7 = lshr <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sr_u8

!----------------------
! vec_srl
!----------------------
! CHECK-LABEL: vec_srl_i1u1
subroutine vec_srl_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u1

! CHECK-LABEL: vec_srl_i2u1
subroutine vec_srl_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u1

! CHECK-LABEL: vec_srl_i4u1
subroutine vec_srl_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_i4u1

! CHECK-LABEL: vec_srl_i1u2
subroutine vec_srl_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u2

! CHECK-LABEL: vec_srl_i2u2
subroutine vec_srl_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u2

! CHECK-LABEL: vec_srl_i4u2
subroutine vec_srl_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_i4u2

! CHECK-LABEL: vec_srl_i1u4
subroutine vec_srl_i1u4(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u4

! CHECK-LABEL: vec_srl_i2u4
subroutine vec_srl_i2u4(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u4

! CHECK-LABEL: vec_srl_i4u4
subroutine vec_srl_i4u4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_srl_i4u4

! CHECK-LABEL: vec_srl_u1u1
subroutine vec_srl_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u1

! CHECK-LABEL: vec_srl_u2u1
subroutine vec_srl_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u1

! CHECK-LABEL: vec_srl_u4u1
subroutine vec_srl_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_u4u1

! CHECK-LABEL: vec_srl_u1u2
subroutine vec_srl_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u2

! CHECK-LABEL: vec_srl_u2u2
subroutine vec_srl_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u2

! CHECK-LABEL: vec_srl_u4u2
subroutine vec_srl_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_u4u2

! CHECK-LABEL: vec_srl_u1u4
subroutine vec_srl_u1u4(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u4

! CHECK-LABEL: vec_srl_u2u4
subroutine vec_srl_u2u4(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u4

! CHECK-LABEL: vec_srl_u4u4
subroutine vec_srl_u4u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_srl_u4u4

!----------------------
! vec_sro
!----------------------

! CHECK-LABEL: vec_sro_i1u1
subroutine vec_sro_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_i1u1

! CHECK-LABEL: vec_sro_i2u1
subroutine vec_sro_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_i2u1

! CHECK-LABEL: vec_sro_i4u1
subroutine vec_sro_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_i4u1

! CHECK-LABEL: vec_sro_u1u1
subroutine vec_sro_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_u1u1

! CHECK-LABEL: vec_sro_u2u1
subroutine vec_sro_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_u2u1

! CHECK-LABEL: vec_sro_u4u1
subroutine vec_sro_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_u4u1

! CHECK-LABEL: vec_sro_r4u1
subroutine vec_sro_r4u1(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_sro_r4u1

!-------------------------------------

! CHECK-LABEL: vec_sro_i1u2
subroutine vec_sro_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_i1u2

! CHECK-LABEL: vec_sro_i2u2
subroutine vec_sro_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_i2u2

! CHECK-LABEL: vec_sro_i4u2
subroutine vec_sro_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_i4u2

! CHECK-LABEL: vec_sro_u1u2
subroutine vec_sro_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_u1u2

! CHECK-LABEL: vec_sro_u2u2
subroutine vec_sro_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>

end subroutine vec_sro_u2u2

! CHECK-LABEL: vec_sro_u4u2
subroutine vec_sro_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_u4u2

! CHECK-LABEL: vec_sro_r4u2
subroutine vec_sro_r4u2(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_sro_r4u2
