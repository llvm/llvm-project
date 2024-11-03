! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! mma_lxvp
!----------------------

      subroutine mma_lxvp_test_i2(v1, offset, vp)
      use, intrinsic :: mma
      integer(2) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      vp = mma_lxvp(offset, v1)
      end subroutine mma_lxvp_test_i2

!CHECK-LABEL: @mma_lxvp_test_i2_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine test_cvspbf16()
      implicit none
      vector(unsigned(1)) :: v1, v2
      v1 = vec_cvspbf16(v2)
      end subroutine test_cvspbf16

!CHECK-LABEL: @test_cvspbf16_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = load <16 x i8>, ptr %2, align 16
!CHECK:  %4 = call <16 x i8> @llvm.ppc.vsx.xvcvspbf16(<16 x i8> %3)
!CHECK:  store <16 x i8> %4, ptr %1, align 16

      subroutine test_cvbf16spn()
      implicit none
      vector(unsigned(1)) :: v1, v2
      v1 = vec_cvbf16spn(v2)
      end subroutine test_cvbf16spn

!CHECK-LABEL: @test_cvbf16spn_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = load <16 x i8>, ptr %2, align 16
!CHECK:  %4 = call <16 x i8> @llvm.ppc.vsx.xvcvbf16spn(<16 x i8> %3)
!CHECK:  store <16 x i8> %4, ptr %1, align 16

!----------------------
! vec_lxvp
!----------------------

      subroutine vec_lxvp_test_i2(v1, offset, vp)
      integer(2) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i2

!CHECK-LABEL: @vec_lxvp_test_i2_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i4(v1, offset, vp)
      integer(2) :: offset
      vector(integer(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i4

!CHECK-LABEL: @vec_lxvp_test_i4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u2(v1, offset, vp)
      integer(2) :: offset
      vector(unsigned(2)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u2

!CHECK-LABEL: @vec_lxvp_test_u2_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u4(v1, offset, vp)
      integer(2) :: offset
      vector(unsigned(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u4

!CHECK-LABEL: @vec_lxvp_test_u4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r4(v1, offset, vp)
      integer(2) :: offset
      vector(real(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r4

!CHECK-LABEL: @vec_lxvp_test_r4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r8(v1, offset, vp)
      integer(2) :: offset
      vector(real(8)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r8

!CHECK-LABEL: @vec_lxvp_test_r8_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_vp(v1, offset, vp)
      integer(2) :: offset
      __vector_pair :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_vp

!CHECK-LABEL: @vec_lxvp_test_vp_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i2_arr(v1, offset, vp)
      integer :: offset
      vector(integer(2)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i2_arr

!CHECK-LABEL: @vec_lxvp_test_i2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i4_arr(v1, offset, vp)
      integer :: offset
      vector(integer(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i4_arr

!CHECK-LABEL: @vec_lxvp_test_i4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u2_arr(v1, offset, vp)
      integer :: offset
      vector(unsigned(2)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u2_arr

!CHECK-LABEL: @vec_lxvp_test_u2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u4_arr(v1, offset, vp)
      integer :: offset
      vector(unsigned(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u4_arr

!CHECK-LABEL: @vec_lxvp_test_u4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r4_arr(v1, offset, vp)
      integer :: offset
      vector(real(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r4_arr

!CHECK-LABEL: @vec_lxvp_test_r4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r8_arr(v1, offset, vp)
      integer :: offset
      vector(real(8)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r8_arr

!CHECK-LABEL: @vec_lxvp_test_r8_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_vp_arr(v1, offset, vp)
      integer(8) :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_vp_arr

!CHECK-LABEL: @vec_lxvp_test_vp_arr_
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

!----------------------
! vsx_lxvp
!----------------------

      subroutine vsx_lxvp_test_i4(v1, offset, vp)
      integer(2) :: offset
      vector(integer(4)) :: v1
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_i4

!CHECK-LABEL: @vsx_lxvp_test_i4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_r8(v1, offset, vp)
      integer(2) :: offset
      vector(real(8)) :: v1
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_r8

!CHECK-LABEL: @vsx_lxvp_test_r8_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_i2_arr(v1, offset, vp)
      integer :: offset
      vector(integer(2)) :: v1(10)
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_i2_arr

!CHECK-LABEL: @vsx_lxvp_test_i2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_vp_arr(v1, offset, vp)
      integer(8) :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_vp_arr

!CHECK-LABEL: @vsx_lxvp_test_vp_arr_
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

!----------------------
! mma_stxvp
!----------------------

      subroutine test_mma_stxvp_i1(vp, offset, v1)
      use, intrinsic :: mma
      integer(1) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      call mma_stxvp(vp, offset, v1)
      end subroutine test_mma_stxvp_i1

!CHECK-LABEL: @test_mma_stxvp_i1_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i8, ptr %1, align 1
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i8 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

!----------------------
! vec_stxvp
!----------------------

      subroutine test_vec_stxvp_i1(vp, offset, v1)
      integer(1) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_i1

!CHECK-LABEL: @test_vec_stxvp_i1_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i8, ptr %1, align 1
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i8 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_i8(vp, offset, v1)
      integer(8) :: offset
      vector(integer(8)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_i8

!CHECK-LABEL: @test_vec_stxvp_i8_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i64 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vi2(vp, offset, v1)
      integer(2) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vi2

!CHECK-LABEL: @test_vec_stxvp_vi2_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vi4(vp, offset, v1)
      integer(2) :: offset
      vector(integer(4)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vi4

!CHECK-LABEL: @test_vec_stxvp_vi4_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vu2(vp, offset, v1)
      integer(2) :: offset
      vector(unsigned(2)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vu2

!CHECK-LABEL: @test_vec_stxvp_vu2_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vu4(vp, offset, v1)
      integer(2) :: offset
      vector(unsigned(4)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vu4

!CHECK-LABEL: @test_vec_stxvp_vu4_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vr4(vp, offset, v1)
      integer(2) :: offset
      vector(real(4)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vr4

!CHECK-LABEL: @test_vec_stxvp_vr4_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vr8(vp, offset, v1)
      integer(2) :: offset
      vector(real(8)) :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vr8

!CHECK-LABEL: @test_vec_stxvp_vr8_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vvp(vp, offset, v1)
      integer(2) :: offset
      __vector_pair :: v1
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vvp

!CHECK-LABEL: @test_vec_stxvp_vvp_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vi2_arr(vp, offset, v1)
      integer :: offset
      vector(integer(2)) :: v1(10)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vi2_arr

!CHECK-LABEL: @test_vec_stxvp_vi2_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vi4_arr(vp, offset, v1)
      integer :: offset
      vector(integer(4)) :: v1(10)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vi4_arr

!CHECK-LABEL: @test_vec_stxvp_vi4_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vu2_arr(vp, offset, v1)
      integer :: offset
      vector(unsigned(2)) :: v1(11)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vu2_arr

!CHECK-LABEL: @test_vec_stxvp_vu2_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vu4_arr(vp, offset, v1)
      integer(8) :: offset
      vector(unsigned(4)) :: v1(11,3)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vu4_arr

!CHECK-LABEL: @test_vec_stxvp_vu4_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i64 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vr4_arr(vp, offset, v1)
      integer :: offset
      vector(real(4)) :: v1(10)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vr4_arr

!CHECK-LABEL: @test_vec_stxvp_vr4_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vr8_arr(vp, offset, v1)
      integer :: offset
      vector(real(8)) :: v1(10)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vr8_arr

!CHECK-LABEL: @test_vec_stxvp_vr8_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vec_stxvp_vp_arr(vp, offset, v1)
      integer :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      call vec_stxvp(vp, offset, v1)
      end subroutine test_vec_stxvp_vp_arr

!CHECK-LABEL: @test_vec_stxvp_vp_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

!----------------------
! vsx_stxvp
!----------------------

      subroutine test_vsx_stxvp_i1(vp, offset, v1)
      integer(1) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      call vsx_stxvp(vp, offset, v1)
      end subroutine test_vsx_stxvp_i1

!CHECK-LABEL: @test_vsx_stxvp_i1_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i8, ptr %1, align 1
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i8 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vsx_stxvp_vi2(vp, offset, v1)
      integer(2) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      call vsx_stxvp(vp, offset, v1)
      end subroutine test_vsx_stxvp_vi2

!CHECK-LABEL: @test_vsx_stxvp_vi2_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i16 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vsx_stxvp_vr8_arr(vp, offset, v1)
      integer :: offset
      vector(real(8)) :: v1(10)
      __vector_pair :: vp
      call vsx_stxvp(vp, offset, v1)
      end subroutine test_vsx_stxvp_vr8_arr

!CHECK-LABEL: @test_vsx_stxvp_vr8_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])

      subroutine test_vsx_stxvp_vp_arr(vp, offset, v1)
      integer :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      call vsx_stxvp(vp, offset, v1)
      end subroutine test_vsx_stxvp_vp_arr

!CHECK-LABEL: @test_vsx_stxvp_vp_arr_
!CHECK:  %[[vp:.*]] = load <256 x i1>, ptr %0, align 32
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[offset]]
!CHECK:  call void @llvm.ppc.vsx.stxvp(<256 x i1> %[[vp]], ptr %[[addr]])
