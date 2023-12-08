! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!---------
! vec_ctf
!---------
! CHECK-LABEL: vec_ctf_test_i4i1
subroutine vec_ctf_test_i4i1(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i1

! CHECK-LABEL: vec_ctf_test_i4i2
subroutine vec_ctf_test_i4i2(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i2

! CHECK-LABEL: vec_ctf_test_i4i4
subroutine vec_ctf_test_i4i4(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i4

! CHECK-LABEL: vec_ctf_test_i4i8
subroutine vec_ctf_test_i4i8(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i8

! CHECK-LABEL: vec_ctf_test_i8i1
subroutine vec_ctf_test_i8i1(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i1

! CHECK-LABEL: vec_ctf_test_i8i2
subroutine vec_ctf_test_i8i2(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i2

! CHECK-LABEL: vec_ctf_test_i8i4
subroutine vec_ctf_test_i8i4(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i4

! CHECK-LABEL: vec_ctf_test_i8i8
subroutine vec_ctf_test_i8i8(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i8

! CHECK-LABEL: vec_ctf_test_u4i1
subroutine vec_ctf_test_u4i1(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i1

! CHECK-LABEL: vec_ctf_test_u4i2
subroutine vec_ctf_test_u4i2(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i2

! CHECK-LABEL: vec_ctf_test_u4i4
subroutine vec_ctf_test_u4i4(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i4

! CHECK-LABEL: vec_ctf_test_u4i8
subroutine vec_ctf_test_u4i8(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i8

! CHECK-LABEL: vec_ctf_test_u8i1
subroutine vec_ctf_test_u8i1(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i1

! CHECK-LABEL: vec_ctf_test_u8i2
subroutine vec_ctf_test_u8i2(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i2

! CHECK-LABEL: vec_ctf_test_u8i4
subroutine vec_ctf_test_u8i4(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i4

! CHECK-LABEL: vec_ctf_test_u8i8
subroutine vec_ctf_test_u8i8(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! LLVMIR: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i8

!-------------
! vec_convert
!-------------
! CHECK-LABEL: vec_convert_test_i1i1
subroutine vec_convert_test_i1i1(v, mold)
  vector(integer(1)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i1

! CHECK-LABEL: vec_convert_test_i1i2
subroutine vec_convert_test_i1i2(v, mold)
  vector(integer(1)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i2

! CHECK-LABEL: vec_convert_test_i1i4
subroutine vec_convert_test_i1i4(v, mold)
  vector(integer(1)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i4

! CHECK-LABEL: vec_convert_test_i1i8
subroutine vec_convert_test_i1i8(v, mold)
  vector(integer(1)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i8

! CHECK-LABEL: vec_convert_test_i1u1
subroutine vec_convert_test_i1u1(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u1

! CHECK-LABEL: vec_convert_test_i1u2
subroutine vec_convert_test_i1u2(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u2

! CHECK-LABEL: vec_convert_test_i1u4
subroutine vec_convert_test_i1u4(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u4

! CHECK-LABEL: vec_convert_test_i1u8
subroutine vec_convert_test_i1u8(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u8

! CHECK-LABEL: vec_convert_test_i1r4
subroutine vec_convert_test_i1r4(v, mold)
  vector(integer(1)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1r4

! CHECK-LABEL: vec_convert_test_i1r8
subroutine vec_convert_test_i1r8(v, mold)
  vector(integer(1)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1r8

! CHECK-LABEL: vec_convert_test_i2i1
subroutine vec_convert_test_i2i1(v, mold)
  vector(integer(2)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i1

! CHECK-LABEL: vec_convert_test_i2i2
subroutine vec_convert_test_i2i2(v, mold)
  vector(integer(2)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i2

! CHECK-LABEL: vec_convert_test_i2i4
subroutine vec_convert_test_i2i4(v, mold)
  vector(integer(2)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i4

! CHECK-LABEL: vec_convert_test_i2i8
subroutine vec_convert_test_i2i8(v, mold)
  vector(integer(2)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i8

! CHECK-LABEL: vec_convert_test_i2u1
subroutine vec_convert_test_i2u1(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u1

! CHECK-LABEL: vec_convert_test_i2u2
subroutine vec_convert_test_i2u2(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u2

! CHECK-LABEL: vec_convert_test_i2u4
subroutine vec_convert_test_i2u4(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u4

! CHECK-LABEL: vec_convert_test_i2u8
subroutine vec_convert_test_i2u8(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u8

! CHECK-LABEL: vec_convert_test_i2r4
subroutine vec_convert_test_i2r4(v, mold)
  vector(integer(2)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2r4

! CHECK-LABEL: vec_convert_test_i2r8
subroutine vec_convert_test_i2r8(v, mold)
  vector(integer(2)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2r8

! CHECK-LABEL: vec_convert_test_i4i1
subroutine vec_convert_test_i4i1(v, mold)
  vector(integer(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i1

! CHECK-LABEL: vec_convert_test_i4i2
subroutine vec_convert_test_i4i2(v, mold)
  vector(integer(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i2

! CHECK-LABEL: vec_convert_test_i4i4
subroutine vec_convert_test_i4i4(v, mold)
  vector(integer(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i4

! CHECK-LABEL: vec_convert_test_i4i8
subroutine vec_convert_test_i4i8(v, mold)
  vector(integer(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i8

! CHECK-LABEL: vec_convert_test_i4u1
subroutine vec_convert_test_i4u1(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u1

! CHECK-LABEL: vec_convert_test_i4u2
subroutine vec_convert_test_i4u2(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u2

! CHECK-LABEL: vec_convert_test_i4u4
subroutine vec_convert_test_i4u4(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u4

! CHECK-LABEL: vec_convert_test_i4u8
subroutine vec_convert_test_i4u8(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u8

! CHECK-LABEL: vec_convert_test_i4r4
subroutine vec_convert_test_i4r4(v, mold)
  vector(integer(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r4

! CHECK-LABEL: vec_convert_test_i4r8
subroutine vec_convert_test_i4r8(v, mold)
  vector(integer(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r8

! CHECK-LABEL: vec_convert_test_i8i1
subroutine vec_convert_test_i8i1(v, mold)
  vector(integer(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i1

! CHECK-LABEL: vec_convert_test_i8i2
subroutine vec_convert_test_i8i2(v, mold)
  vector(integer(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i2

! CHECK-LABEL: vec_convert_test_i8i4
subroutine vec_convert_test_i8i4(v, mold)
  vector(integer(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i4

! CHECK-LABEL: vec_convert_test_i8i8
subroutine vec_convert_test_i8i8(v, mold)
  vector(integer(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i8

! CHECK-LABEL: vec_convert_test_i8u1
subroutine vec_convert_test_i8u1(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u1

! CHECK-LABEL: vec_convert_test_i8u2
subroutine vec_convert_test_i8u2(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u2

! CHECK-LABEL: vec_convert_test_i8u4
subroutine vec_convert_test_i8u4(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u4

! CHECK-LABEL: vec_convert_test_i8u8
subroutine vec_convert_test_i8u8(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u8

! CHECK-LABEL: vec_convert_test_i8r4
subroutine vec_convert_test_i8r4(v, mold)
  vector(integer(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8r4

! CHECK-LABEL: vec_convert_test_i8r8
subroutine vec_convert_test_i8r8(v, mold)
  vector(integer(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8r8

! CHECK-LABEL: vec_convert_test_u1i1
subroutine vec_convert_test_u1i1(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i1

! CHECK-LABEL: vec_convert_test_u1i2
subroutine vec_convert_test_u1i2(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i2

! CHECK-LABEL: vec_convert_test_u1i4
subroutine vec_convert_test_u1i4(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i4

! CHECK-LABEL: vec_convert_test_u1i8
subroutine vec_convert_test_u1i8(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i8

! CHECK-LABEL: vec_convert_test_u1u1
subroutine vec_convert_test_u1u1(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u1

! CHECK-LABEL: vec_convert_test_u1u2
subroutine vec_convert_test_u1u2(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u2

! CHECK-LABEL: vec_convert_test_u1u4
subroutine vec_convert_test_u1u4(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u4

! CHECK-LABEL: vec_convert_test_u1u8
subroutine vec_convert_test_u1u8(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u8

! CHECK-LABEL: vec_convert_test_u1r4
subroutine vec_convert_test_u1r4(v, mold)
  vector(unsigned(1)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1r4

! CHECK-LABEL: vec_convert_test_u1r8
subroutine vec_convert_test_u1r8(v, mold)
  vector(unsigned(1)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1r8

! CHECK-LABEL: vec_convert_test_u2i1
subroutine vec_convert_test_u2i1(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i1

! CHECK-LABEL: vec_convert_test_u2i2
subroutine vec_convert_test_u2i2(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i2

! CHECK-LABEL: vec_convert_test_u2i4
subroutine vec_convert_test_u2i4(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i4

! CHECK-LABEL: vec_convert_test_u2i8
subroutine vec_convert_test_u2i8(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i8

! CHECK-LABEL: vec_convert_test_u2u1
subroutine vec_convert_test_u2u1(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u1

! CHECK-LABEL: vec_convert_test_u2u2
subroutine vec_convert_test_u2u2(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u2

! CHECK-LABEL: vec_convert_test_u2u4
subroutine vec_convert_test_u2u4(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u4

! CHECK-LABEL: vec_convert_test_u2u8
subroutine vec_convert_test_u2u8(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u8

! CHECK-LABEL: vec_convert_test_u2r4
subroutine vec_convert_test_u2r4(v, mold)
  vector(unsigned(2)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2r4

! CHECK-LABEL: vec_convert_test_u2r8
subroutine vec_convert_test_u2r8(v, mold)
  vector(unsigned(2)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2r8

! CHECK-LABEL: vec_convert_test_u4i1
subroutine vec_convert_test_u4i1(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i1

! CHECK-LABEL: vec_convert_test_u4i2
subroutine vec_convert_test_u4i2(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i2

! CHECK-LABEL: vec_convert_test_u4i4
subroutine vec_convert_test_u4i4(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i4

! CHECK-LABEL: vec_convert_test_u4i8
subroutine vec_convert_test_u4i8(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i8

! CHECK-LABEL: vec_convert_test_u4u1
subroutine vec_convert_test_u4u1(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u1

! CHECK-LABEL: vec_convert_test_u4u2
subroutine vec_convert_test_u4u2(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u2

! CHECK-LABEL: vec_convert_test_u4u4
subroutine vec_convert_test_u4u4(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u4

! CHECK-LABEL: vec_convert_test_u4u8
subroutine vec_convert_test_u4u8(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u8

! CHECK-LABEL: vec_convert_test_u4r4
subroutine vec_convert_test_u4r4(v, mold)
  vector(unsigned(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4r4

! CHECK-LABEL: vec_convert_test_u4r8
subroutine vec_convert_test_u4r8(v, mold)
  vector(unsigned(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4r8

! CHECK-LABEL: vec_convert_test_u8i1
subroutine vec_convert_test_u8i1(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i1

! CHECK-LABEL: vec_convert_test_u8i2
subroutine vec_convert_test_u8i2(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i2

! CHECK-LABEL: vec_convert_test_u8i4
subroutine vec_convert_test_u8i4(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i4

! CHECK-LABEL: vec_convert_test_u8i8
subroutine vec_convert_test_u8i8(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i8

! CHECK-LABEL: vec_convert_test_u8u1
subroutine vec_convert_test_u8u1(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u1

! CHECK-LABEL: vec_convert_test_u8u2
subroutine vec_convert_test_u8u2(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u2

! CHECK-LABEL: vec_convert_test_u8u4
subroutine vec_convert_test_u8u4(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u4

! CHECK-LABEL: vec_convert_test_u8u8
subroutine vec_convert_test_u8u8(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u8

! CHECK-LABEL: vec_convert_test_u8r4
subroutine vec_convert_test_u8r4(v, mold)
  vector(unsigned(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8r4

! CHECK-LABEL: vec_convert_test_u8r8
subroutine vec_convert_test_u8r8(v, mold)
  vector(unsigned(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8r8

! CHECK-LABEL: vec_convert_test_r4i1
subroutine vec_convert_test_r4i1(v, mold)
  vector(real(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i1

! CHECK-LABEL: vec_convert_test_r4i2
subroutine vec_convert_test_r4i2(v, mold)
  vector(real(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i2

! CHECK-LABEL: vec_convert_test_r4i4
subroutine vec_convert_test_r4i4(v, mold)
  vector(real(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i4

! CHECK-LABEL: vec_convert_test_r4i8
subroutine vec_convert_test_r4i8(v, mold)
  vector(real(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i8

! CHECK-LABEL: vec_convert_test_r4u1
subroutine vec_convert_test_r4u1(v, mold)
  vector(real(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u1

! CHECK-LABEL: vec_convert_test_r4u2
subroutine vec_convert_test_r4u2(v, mold)
  vector(real(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u2

! CHECK-LABEL: vec_convert_test_r4u4
subroutine vec_convert_test_r4u4(v, mold)
  vector(real(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u4

! CHECK-LABEL: vec_convert_test_r4u8
subroutine vec_convert_test_r4u8(v, mold)
  vector(real(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u8

! CHECK-LABEL: vec_convert_test_r4r4
subroutine vec_convert_test_r4r4(v, mold)
  vector(real(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: store <4 x float> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4r4

! CHECK-LABEL: vec_convert_test_r4r8
subroutine vec_convert_test_r4r8(v, mold)
  vector(real(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4r8

! CHECK-LABEL: vec_convert_test_r8i1
subroutine vec_convert_test_r8i1(v, mold)
  vector(real(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i1

! CHECK-LABEL: vec_convert_test_r8i2
subroutine vec_convert_test_r8i2(v, mold)
  vector(real(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i2

! CHECK-LABEL: vec_convert_test_r8i4
subroutine vec_convert_test_r8i4(v, mold)
  vector(real(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i4

! CHECK-LABEL: vec_convert_test_r8i8
subroutine vec_convert_test_r8i8(v, mold)
  vector(real(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i8

! CHECK-LABEL: vec_convert_test_r8u1
subroutine vec_convert_test_r8u1(v, mold)
  vector(real(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u1

! CHECK-LABEL: vec_convert_test_r8u2
subroutine vec_convert_test_r8u2(v, mold)
  vector(real(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u2

! CHECK-LABEL: vec_convert_test_r8u4
subroutine vec_convert_test_r8u4(v, mold)
  vector(real(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u4

! CHECK-LABEL: vec_convert_test_r8u8
subroutine vec_convert_test_r8u8(v, mold)
  vector(real(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u8

! CHECK-LABEL: vec_convert_test_r8r4
subroutine vec_convert_test_r8r4(v, mold)
  vector(real(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8r4

! CHECK-LABEL: vec_convert_test_r8r8
subroutine vec_convert_test_r8r8(v, mold)
  vector(real(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! LLVMIR: store <2 x double> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8r8

! CHECK-LABEL: vec_convert_test_i1i1_array
subroutine vec_convert_test_i1i1_array(v, mold)
  vector(integer(1)) :: v
  vector(integer(1)) :: mold(4, 8), r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i1_array

! CHECK-LABEL: vec_convert_test_i4r8_array
subroutine vec_convert_test_i4r8_array(v, mold)
  vector(integer(4)) :: v
  vector(real(8)) :: mold(2, 4, 8), r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r8_array

! CHECK-LABEL: vec_convert_test_u8i2_array
subroutine vec_convert_test_u8i2_array(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(2)) :: mold(10), r
  r = vec_convert(v, mold)

! LLVMIR: %[[v:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i2_array

!---------
! vec_cvf
!---------
! CHECK-LABEL: vec_cvf_test_r4r8
subroutine vec_cvf_test_r4r8(arg1)
  vector(real(8)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_cvf(arg1)

! LLVMIR: %[[arg:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[call:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvcvdpsp(<2 x double> %[[arg]])
! LLVMIR: %[[b:.*]] = bitcast <4 x float> %[[call]] to <16 x i8>
! LLVMIR: %[[sh:.*]] = shufflevector <16 x i8> %[[b]], <16 x i8> %[[b]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11>
! LLVMIR: %[[r:.*]] = bitcast <16 x i8> %[[sh]] to <4 x float>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r4r8

! CHECK-LABEL: vec_cvf_test_r8r4
subroutine vec_cvf_test_r8r4(arg1)
  vector(real(4)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_cvf(arg1)

! LLVMIR: %[[arg:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[bfi:.*]] = bitcast <4 x float> %[[arg]] to <16 x i8>
! LLVMIR: %[[sh:.*]] = shufflevector <16 x i8> %[[bfi]], <16 x i8> %[[bfi]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11>
! LLVMIR: %[[bif:.*]] = bitcast <16 x i8> %[[sh]] to <4 x float>
! LLVMIR: %[[r:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvcvspdp(<4 x float> %[[bif]])
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r8r4

