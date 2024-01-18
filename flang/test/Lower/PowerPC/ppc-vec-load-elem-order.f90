! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!-------------------
! vec_ld
!-------------------

! CHECK-LABEL: @vec_ld_testi8
subroutine vec_ld_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[bc]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi8

! CHECK-LABEL: @vec_ld_testi16
subroutine vec_ld_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[bc]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi16

! CHECK-LABEL: @vec_ld_testi32
subroutine vec_ld_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32

! CHECK-LABEL: @vec_ld_testf32
subroutine vec_ld_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[i4:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[i4]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testf32

! CHECK-LABEL: @vec_ld_testu32
subroutine vec_ld_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testu32

! CHECK-LABEL: @vec_ld_testi32a
subroutine vec_ld_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32a

! CHECK-LABEL: @vec_ld_testf32av
subroutine vec_ld_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[i4:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[i4]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testf32av

! CHECK-LABEL: @vec_ld_testi32s
subroutine vec_ld_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32s

!-------------------
! vec_lde
!-------------------

! CHECK-LABEL: @vec_lde_testi8s
subroutine vec_lde_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvebx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi8s

! CHECK-LABEL: @vec_lde_testi16a
subroutine vec_lde_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 11, 7)
  vector(integer(2)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <8 x i16> @llvm.ppc.altivec.lvehx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[ld]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi16a

! CHECK-LABEL: @vec_lde_testi32a
subroutine vec_lde_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(5)
  vector(integer(4)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi32a

! CHECK-LABEL: @vec_lde_testf32a
subroutine vec_lde_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(11)
  vector(real(4)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testf32a

!-------------------
! vec_lvsl
!-------------------

! CHECK-LABEL: @vec_lvsl_testi8s
subroutine vec_lvsl_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[iext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi8s

! CHECK-LABEL: @vec_lvsl_testi16a
subroutine vec_lvsl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[iext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi16a

! CHECK-LABEL: @vec_lvsl_testi32a
subroutine vec_lvsl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(11, 3, 4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi32a

! CHECK-LABEL: @vec_lvsl_testf32a
subroutine vec_lvsl_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(51)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testf32a

!-------------------
! vec_lvsr
!-------------------

! CHECK-LABEL: @vec_lvsr_testi8s
subroutine vec_lvsr_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[iext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi8s

! CHECK-LABEL: @vec_lvsr_testi16a
subroutine vec_lvsr_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(41)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[iext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi16a

! CHECK-LABEL: @vec_lvsr_testi32a
subroutine vec_lvsr_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(23, 31, 47)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi32a

! CHECK-LABEL: @vec_lvsr_testf32a
subroutine vec_lvsr_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testf32a

!-------------------
! vec_lxv
!-------------------

! CHECK-LABEL: @vec_lxv_testi8a
subroutine vec_lxv_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR: %[[offset:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[offset]]
! LLVMIR: %[[res:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: store <16 x i8> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi8a

! CHECK-LABEL: @vec_lxv_testi16a
subroutine vec_lxv_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR: %[[offset:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[offset]]
! LLVMIR: %[[res:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: store <8 x i16> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi16a

! CHECK-LABEL: @vec_lxv_testi32a
subroutine vec_lxv_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR: %[[offset:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[offset]]
! LLVMIR: %[[res:.*]] = load <4 x i32>, ptr %[[addr]], align 1
! LLVMIR: store <4 x i32> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi32a

! CHECK-LABEL: @vec_lxv_testf32a
subroutine vec_lxv_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR: %[[offset:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[offset]]
! LLVMIR: %[[res:.*]] = load <4 x float>, ptr %[[addr]], align 1
! LLVMIR: store <4 x float> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testf32a

! CHECK-LABEL: @vec_lxv_testf64a
subroutine vec_lxv_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(4)
  vector(real(8)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR: %[[offset:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[offset]]
! LLVMIR: %[[res:.*]] = load <2 x double>, ptr %[[addr]], align 1
! LLVMIR: store <2 x double> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testf64a

!-------------------
! vec_xl
!-------------------

! CHECK-LABEL: @vec_xl_testi8a
subroutine vec_xl_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_xl(arg1, arg2)

  
! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_xl_testi8a

! CHECK-LABEL: @vec_xl_testi16a
subroutine vec_xl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 8)
  vector(integer(2)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[ld]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_xl_testi16a

! CHECK-LABEL: @vec_xl_testi32a
subroutine vec_xl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testi32a

! CHECK-LABEL: @vec_xl_testi64a
subroutine vec_xl_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 1)
  vector(integer(8)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xl_testi64a

! CHECK-LABEL: @vec_xl_testf32a
subroutine vec_xl_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xl_testf32a

! CHECK-LABEL: @vec_xl_testf64a
subroutine vec_xl_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(2)
  vector(real(8)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testf64a

!-------------------
! vec_xl_be
!-------------------

! CHECK-LABEL: @vec_xl_be_testi8a
subroutine vec_xl_be_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(2, 4, 8)
  vector(integer(1)) :: res
  res = vec_xl_be(arg1, arg2)

  
! LLVMIR: %4 = load i8, ptr %0, align 1
! LLVMIR: %5 = getelementptr i8, ptr %1, i8 %4
! LLVMIR: %6 = load <16 x i8>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %7, ptr %2, align 16
end subroutine vec_xl_be_testi8a

! CHECK-LABEL: @vec_xl_be_testi16a
subroutine vec_xl_be_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(8,2)
  vector(integer(2)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %4 = load i16, ptr %0, align 2
! LLVMIR: %5 = getelementptr i8, ptr %1, i16 %4
! LLVMIR: %6 = load <8 x i16>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <8 x i16> %6, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %7, ptr %2, align 16
end subroutine vec_xl_be_testi16a

! CHECK-LABEL: @vec_xl_be_testi32a
subroutine vec_xl_be_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4)
  vector(integer(4)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %4 = load i32, ptr %0, align 4
! LLVMIR: %5 = getelementptr i8, ptr %1, i32 %4
! LLVMIR: %6 = load <4 x i32>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <4 x i32> %6, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %7, ptr %2, align 16
end subroutine vec_xl_be_testi32a

! CHECK-LABEL: @vec_xl_be_testi64a
subroutine vec_xl_be_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 8)
  vector(integer(8)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %4 = load i64, ptr %0, align 8
! LLVMIR: %5 = getelementptr i8, ptr %1, i64 %4
! LLVMIR: %6 = load <2 x i64>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <2 x i64> %6, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x i64> %7, ptr %2, align 16
end subroutine vec_xl_be_testi64a

! CHECK-LABEL: @vec_xl_be_testf32a
subroutine vec_xl_be_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %4 = load i16, ptr %0, align 2
! LLVMIR: %5 = getelementptr i8, ptr %1, i16 %4
! LLVMIR: %6 = load <4 x float>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <4 x float> %6, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %7, ptr %2, align 16
end subroutine vec_xl_be_testf32a

! CHECK-LABEL: @vec_xl_be_testf64a
subroutine vec_xl_be_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %4 = load i64, ptr %0, align 8
! LLVMIR: %5 = getelementptr i8, ptr %1, i64 %4
! LLVMIR: %6 = load <2 x double>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <2 x double> %6, <2 x double> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x double> %7, ptr %2, align 16
end subroutine vec_xl_be_testf64a

!-------------------
! vec_xld2
!-------------------

! CHECK-LABEL: @vec_xld2_testi8a
subroutine vec_xld2_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi8a

! CHECK-LABEL: @vec_xld2_testi16a
subroutine vec_xld2_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(4)
  vector(integer(2)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <8 x i16>
! LLVMIR:  store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi16a

! CHECK-LABEL: @vec_xld2_testi32a
subroutine vec_xld2_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2(11)
  vector(integer(4)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi32a

! CHECK-LABEL: @vec_xld2_testi64a
subroutine vec_xld2_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(31,7)
  vector(integer(8)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi64a

! CHECK-LABEL: @vec_xld2_testf32a
subroutine vec_xld2_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(5)
  vector(real(4)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testf32a

! CHECK-LABEL: @vec_xld2_testf64a
subroutine vec_xld2_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(8)) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xld2_testf64a

!-------------------
! vec_xlw4
!-------------------

! CHECK-LABEL: @vec_xlw4_testi8a
subroutine vec_xlw4_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(2, 11, 37)
  vector(integer(1)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testi8a

! CHECK-LABEL: @vec_xlw4_testi16a
subroutine vec_xlw4_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(2, 8)
  vector(integer(2)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testi16a

! CHECK-LABEL: @vec_xlw4_testu32a
subroutine vec_xlw4_testu32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(unsigned(4)) :: arg2(8, 4)
  vector(unsigned(4)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xlw4_testu32a

! CHECK-LABEL: @vec_xlw4_testf32a
subroutine vec_xlw4_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2
  vector(real(4)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testf32a

!-------------------
! vec_xlds
!-------------------

! CHECK-LABEL: @vec_xlds_testi64a
subroutine vec_xlds_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(4)
  vector(integer(8)) :: res
  res = vec_xlds(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load i64, ptr %[[addr]], align 8
! LLVMIR: %[[insrt:.*]] = insertelement <2 x i64> undef, i64 %[[ld]], i32 0
! LLVMIR: %[[shflv:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[shflv]], ptr %2, align 16
end subroutine vec_xlds_testi64a

! CHECK-LABEL: @vec_xlds_testf64a
subroutine vec_xlds_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(8)) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xlds(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load i64, ptr %[[addr]], align 8
! LLVMIR: %[[insrt:.*]] = insertelement <2 x i64> undef, i64 %[[ld]], i32 0
! LLVMIR: %[[shflv:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: %[[bc:.*]] = bitcast <2 x i64> %[[shflv]] to <2 x double>
! LLVMIR: store <2 x double> %[[bc]], ptr %2, align 16
end subroutine vec_xlds_testf64a
