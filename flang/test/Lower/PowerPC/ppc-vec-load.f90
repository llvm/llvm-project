! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_ld
!----------------------

! CHECK-LABEL: @vec_ld_testi8
subroutine vec_ld_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %{{.*}}, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi8

! CHECK-LABEL: @vec_ld_testi16
subroutine vec_ld_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi16

! CHECK-LABEL: @vec_ld_testi32
subroutine vec_ld_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[bc:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi32

! CHECK-LABEL: @vec_ld_testf32
subroutine vec_ld_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[arg1i32:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1i32]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testf32

! CHECK-LABEL: @vec_ld_testu32
subroutine vec_ld_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ld_testu32

! CHECK-LABEL: @vec_ld_testi32a
subroutine vec_ld_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ld_testi32a

! CHECK-LABEL: @vec_ld_testf32av
subroutine vec_ld_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[arg1i32:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1i32]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testf32av

! CHECK-LABEL: @vec_ld_testi32s
subroutine vec_ld_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi32s

!----------------------
! vec_lde
!----------------------

! CHECK-LABEL: @vec_lde_testi8s
subroutine vec_lde_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <16 x i8> @llvm.ppc.altivec.lvebx(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi8s

! CHECK-LABEL: @vec_lde_testi16a
subroutine vec_lde_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <8 x i16> @llvm.ppc.altivec.lvehx(ptr %[[addr]])
! LLVMIR: store <8 x i16> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi16a

! CHECK-LABEL: @vec_lde_testi32a
subroutine vec_lde_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(4)
  vector(integer(4)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi32a

! CHECK-LABEL: @vec_lde_testf32a
subroutine vec_lde_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_lde(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_lde_testf32a

!----------------------
! vec_ldl
!----------------------

! CHECK-LABEL: @vec_ldl_testi8
subroutine vec_ldl_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %{{.*}}, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi8

! CHECK-LABEL: @vec_ldl_testi16
subroutine vec_ldl_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi16

! CHECK-LABEL: @vec_ldl_testi32
subroutine vec_ldl_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[bc:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi32

! CHECK-LABEL: @vec_ldl_testf32
subroutine vec_ldl_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)


! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testf32

! CHECK-LABEL: @vec_ldl_testu32
subroutine vec_ldl_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ldl_testu32

! CHECK-LABEL: @vec_ldl_testi32a
subroutine vec_ldl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ldl_testi32a

! CHECK-LABEL: @vec_ldl_testf32av
subroutine vec_ldl_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testf32av

! CHECK-LABEL: @vec_ldl_testi32s
subroutine vec_ldl_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ldl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi32s

!----------------------
! vec_lvsl
!----------------------

! CHECK-LABEL: @vec_lvsl_testi8s
subroutine vec_lvsl_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[ext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsl_testi8s

! CHECK-LABEL: @vec_lvsl_testi16a
subroutine vec_lvsl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[ext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsl_testi16a

! CHECK-LABEL: @vec_lvsl_testi32a
subroutine vec_lvsl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 3, 4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[ext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsl_testi32a

! CHECK-LABEL: @vec_lvsl_testf32a
subroutine vec_lvsl_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsl_testf32a

!----------------------
! vec_lvsr
!----------------------

! CHECK-LABEL: @vec_lvsr_testi8s
subroutine vec_lvsr_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[ext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[ld:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[addr:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[ld]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[addr]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsr_testi8s

! CHECK-LABEL: @vec_lvsr_testi16a
subroutine vec_lvsr_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[ext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[ld:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[addr:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[ld]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[addr]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsr_testi16a

! CHECK-LABEL: @vec_lvsr_testi32a
subroutine vec_lvsr_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 3, 4)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[ext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[ext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[ld:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[addr:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[ld]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[addr]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsr_testi32a

! CHECK-LABEL: @vec_lvsr_testf32a
subroutine vec_lvsr_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: %[[sv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[sv]], ptr %2, align 16
end subroutine vec_lvsr_testf32a

!----------------------
! vec_lxv
!----------------------

! CHECK-LABEL: @vec_lxv_testi8a
subroutine vec_lxv_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR_P9: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR_P9: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR_P9: %[[ld:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR_P9: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lxv_testi8a

! CHECK-LABEL: @vec_lxv_testi16a
subroutine vec_lxv_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR_P9: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR_P9: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR_P9: %[[ld:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR_P9: store <8 x i16> %[[ld]], ptr %2, align 16
end subroutine vec_lxv_testi16a

! CHECK-LABEL: @vec_lxv_testi32a
subroutine vec_lxv_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR_P9: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR_P9: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR_P9: %[[ld:.*]] = load <4 x i32>, ptr %[[addr]], align 1
! LLVMIR_P9: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_lxv_testi32a

! CHECK-LABEL: @vec_lxv_testf32a
subroutine vec_lxv_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR_P9: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR_P9: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR_P9: %[[ld:.*]] = load <4 x float>, ptr %[[addr]], align 1
! LLVMIR_P9: store <4 x float> %[[ld]], ptr %2, align 16
end subroutine vec_lxv_testf32a

! CHECK-LABEL: @vec_lxv_testf64a
subroutine vec_lxv_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(4)
  vector(real(8)) :: res
  res = vec_lxv(arg1, arg2)

! LLVMIR_P9: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR_P9: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR_P9: %[[ld:.*]] = load <2 x double>, ptr %[[addr]], align 1
! LLVMIR_P9: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_lxv_testf64a

!----------------------
! vec_xld2
!----------------------

! CHECK-LABEL: @vec_xld2_testi8a
subroutine vec_xld2_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi8a

! CHECK-LABEL: @vec_xld2_testi16
subroutine vec_xld2_testi16(arg1, arg2, res)
  integer :: arg1
  vector(integer(2)) :: arg2
  vector(integer(2)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi16

! CHECK-LABEL: @vec_xld2_testi32a
subroutine vec_xld2_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2(41)
  vector(integer(4)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi32a

! CHECK-LABEL: @vec_xld2_testi64a
subroutine vec_xld2_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(4)
  vector(integer(8)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi64a

! CHECK-LABEL: @vec_xld2_testf32a
subroutine vec_xld2_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xld2(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
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
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xld2_testf64a

!----------------------
! vec_xl
!----------------------

! CHECK-LABEL: @vec_xl_testi8a
subroutine vec_xl_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testi8a

! CHECK-LABEL: @vec_xl_testi16a
subroutine vec_xl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: store <8 x i16> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testi16a

! CHECK-LABEL: @vec_xl_testi32a
subroutine vec_xl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testi32a

! CHECK-LABEL: @vec_xl_testi64a
subroutine vec_xl_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 8)
  vector(integer(8)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
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
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xl_testf32a

! CHECK-LABEL: @vec_xl_testf64a
subroutine vec_xl_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2
  vector(real(8)) :: res
  res = vec_xl(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testf64a

!----------------------
! vec_xlds
!----------------------

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
! LLVMIR: %[[shfl:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[shfl]], ptr %2, align 16
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
! LLVMIR: %[[shfl:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: %[[bc:.*]] = bitcast <2 x i64> %[[shfl]] to <2 x double>
! LLVMIR: store <2 x double> %[[bc]], ptr %2, align 16
end subroutine vec_xlds_testf64a

!----------------------
! vec_xl_be
!----------------------

! CHECK-LABEL: @vec_xl_be_testi8a
subroutine vec_xl_be_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(2, 4, 8)
  vector(integer(1)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: %[[shff:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testi8a

! CHECK-LABEL: @vec_xl_be_testi16a
subroutine vec_xl_be_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: %[[shff:.*]] = shufflevector <8 x i16> %[[ld]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testi16a

! CHECK-LABEL: @vec_xl_be_testi32a
subroutine vec_xl_be_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR:  %[[ld:.*]] = load <4 x i32>, ptr %[[addr]], align 1
! LLVMIR:  %[[shff:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <4 x i32> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testi32a

! CHECK-LABEL: @vec_xl_be_testi64a
subroutine vec_xl_be_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 8)
  vector(integer(8)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR:  %[[ld:.*]] = load <2 x i64>, ptr %[[addr]], align 1
! LLVMIR:  %[[shff:.*]] = shufflevector <2 x i64> %[[ld]], <2 x i64> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR:  store <2 x i64> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testi64a

! CHECK-LABEL: @vec_xl_be_testf32a
subroutine vec_xl_be_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR:  %[[ld:.*]] = load <4 x float>, ptr %[[addr]], align 1
! LLVMIR:  %[[shff:.*]] = shufflevector <4 x float> %[[ld]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <4 x float> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testf32a

! CHECK-LABEL: @vec_xl_be_testf64a
subroutine vec_xl_be_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(7)
  vector(real(8)) :: res
  res = vec_xl_be(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR:  %[[ld:.*]] = load <2 x double>, ptr %[[addr]], align 1
! LLVMIR:  %[[shff:.*]] = shufflevector <2 x double> %[[ld]], <2 x double> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR:  store <2 x double> %[[shff]], ptr %2, align 16
end subroutine vec_xl_be_testf64a

!----------------------
! vec_xlw4
!----------------------

! CHECK-LABEL: @vec_xlw4_testi8a
subroutine vec_xlw4_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(2, 4, 8)
  vector(integer(1)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testi8a

! CHECK-LABEL: @vec_xlw4_testi16a
subroutine vec_xlw4_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testi16a

! CHECK-LABEL: @vec_xlw4_testu32a
subroutine vec_xlw4_testu32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(unsigned(4)) :: arg2(2, 4, 8)
  vector(unsigned(4)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xlw4_testu32a

! CHECK-LABEL: @vec_xlw4_testf32a
subroutine vec_xlw4_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xlw4(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testf32a
