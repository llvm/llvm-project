! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

! mma_assemble_acc

      subroutine test_assemble_acc_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i1

! CHECK-LABEL: @test_assemble_acc_i1
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %4 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %5 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %6 = load <16 x i8>, ptr %2, align 16
! CHECK:  %7 = load <16 x i8>, ptr %3, align 16
! CHECK:  %8 = load <16 x i8>, ptr %4, align 16
! CHECK:  %9 = load <16 x i8>, ptr %5, align 16
! CHECK:  %10 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %6, <16 x i8> %7, <16 x i8> %8, <16 x i8> %9)
! CHECK:  store <512 x i1> %10, ptr %1, align 64

      subroutine test_assemble_acc_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i2

! CHECK-LABEL: @test_assemble_acc_i2
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %3 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %4 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %5 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %6 = load <8 x i16>, ptr %2, align 16
! CHECK:  %7 = load <8 x i16>, ptr %3, align 16
! CHECK:  %8 = load <8 x i16>, ptr %4, align 16
! CHECK:  %9 = load <8 x i16>, ptr %5, align 16
! CHECK:  %10 = bitcast <8 x i16> %6 to <16 x i8>
! CHECK:  %11 = bitcast <8 x i16> %7 to <16 x i8>
! CHECK:  %12 = bitcast <8 x i16> %8 to <16 x i8>
! CHECK:  %13 = bitcast <8 x i16> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64


      subroutine test_assemble_acc_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i4

! CHECK-LABEL: @test_assemble_acc_i4
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %3 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %4 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %5 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %6 = load <4 x i32>, ptr %2, align 16
! CHECK:  %7 = load <4 x i32>, ptr %3, align 16
! CHECK:  %8 = load <4 x i32>, ptr %4, align 16
! CHECK:  %9 = load <4 x i32>, ptr %5, align 16
! CHECK:  %10 = bitcast <4 x i32> %6 to <16 x i8>
! CHECK:  %11 = bitcast <4 x i32> %7 to <16 x i8>
! CHECK:  %12 = bitcast <4 x i32> %8 to <16 x i8>
! CHECK:  %13 = bitcast <4 x i32> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_assemble_acc_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i8

! CHECK-LABEL: @test_assemble_acc_i8
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %3 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %4 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %5 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %6 = load <2 x i64>, ptr %2, align 16
! CHECK:  %7 = load <2 x i64>, ptr %3, align 16
! CHECK:  %8 = load <2 x i64>, ptr %4, align 16
! CHECK:  %9 = load <2 x i64>, ptr %5, align 16
! CHECK:  %10 = bitcast <2 x i64> %6 to <16 x i8>
! CHECK:  %11 = bitcast <2 x i64> %7 to <16 x i8>
! CHECK:  %12 = bitcast <2 x i64> %8 to <16 x i8>
! CHECK:  %13 = bitcast <2 x i64> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64


      subroutine test_assemble_acc_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u1

! CHECK-LABEL: @test_assemble_acc_u1
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %4 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %5 = alloca <16 x i8>, i64 1, align 16
! CHECK:  %6 = load <16 x i8>, ptr %2, align 16
! CHECK:  %7 = load <16 x i8>, ptr %3, align 16
! CHECK:  %8 = load <16 x i8>, ptr %4, align 16
! CHECK:  %9 = load <16 x i8>, ptr %5, align 16
! CHECK:  %10 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %6, <16 x i8> %7, <16 x i8> %8, <16 x i8> %9)
! CHECK:  store <512 x i1> %10, ptr %1, align 64

      subroutine test_assemble_acc_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u2

! CHECK-LABEL: @test_assemble_acc_u2
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %3 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %4 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %5 = alloca <8 x i16>, i64 1, align 16
! CHECK:  %6 = load <8 x i16>, ptr %2, align 16
! CHECK:  %7 = load <8 x i16>, ptr %3, align 16
! CHECK:  %8 = load <8 x i16>, ptr %4, align 16
! CHECK:  %9 = load <8 x i16>, ptr %5, align 16
! CHECK:  %10 = bitcast <8 x i16> %6 to <16 x i8>
! CHECK:  %11 = bitcast <8 x i16> %7 to <16 x i8>
! CHECK:  %12 = bitcast <8 x i16> %8 to <16 x i8>
! CHECK:  %13 = bitcast <8 x i16> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_assemble_acc_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u4

! CHECK-LABEL: @test_assemble_acc_u4
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %3 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %4 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %5 = alloca <4 x i32>, i64 1, align 16
! CHECK:  %6 = load <4 x i32>, ptr %2, align 16
! CHECK:  %7 = load <4 x i32>, ptr %3, align 16
! CHECK:  %8 = load <4 x i32>, ptr %4, align 16
! CHECK:  %9 = load <4 x i32>, ptr %5, align 16
! CHECK:  %10 = bitcast <4 x i32> %6 to <16 x i8>
! CHECK:  %11 = bitcast <4 x i32> %7 to <16 x i8>
! CHECK:  %12 = bitcast <4 x i32> %8 to <16 x i8>
! CHECK:  %13 = bitcast <4 x i32> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_assemble_acc_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u8

! CHECK-LABEL: @test_assemble_acc_u8
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %3 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %4 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %5 = alloca <2 x i64>, i64 1, align 16
! CHECK:  %6 = load <2 x i64>, ptr %2, align 16
! CHECK:  %7 = load <2 x i64>, ptr %3, align 16
! CHECK:  %8 = load <2 x i64>, ptr %4, align 16
! CHECK:  %9 = load <2 x i64>, ptr %5, align 16
! CHECK:  %10 = bitcast <2 x i64> %6 to <16 x i8>
! CHECK:  %11 = bitcast <2 x i64> %7 to <16 x i8>
! CHECK:  %12 = bitcast <2 x i64> %8 to <16 x i8>
! CHECK:  %13 = bitcast <2 x i64> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_assemble_acc_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_r4

! CHECK-LABEL: @test_assemble_acc_r4
! CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
! CHECK:  %2 = alloca <4 x float>, i64 1, align 16
! CHECK:  %3 = alloca <4 x float>, i64 1, align 16
! CHECK:  %4 = alloca <4 x float>, i64 1, align 16
! CHECK:  %5 = alloca <4 x float>, i64 1, align 16
! CHECK:  %6 = load <4 x float>, ptr %2, align 16
! CHECK:  %7 = load <4 x float>, ptr %3, align 16
! CHECK:  %8 = load <4 x float>, ptr %4, align 16
! CHECK:  %9 = load <4 x float>, ptr %5, align 16
! CHECK:  %10 = bitcast <4 x float> %6 to <16 x i8>
! CHECK:  %11 = bitcast <4 x float> %7 to <16 x i8>
! CHECK:  %12 = bitcast <4 x float> %8 to <16 x i8>
! CHECK:  %13 = bitcast <4 x float> %9 to <16 x i8>
! CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
! CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_assemble_acc_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_r8

!CHECK-LABEL: @test_assemble_acc_r8
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <2 x double>, i64 1, align 16
!CHECK:   %3 = alloca <2 x double>, i64 1, align 16
!CHECK:   %4 = alloca <2 x double>, i64 1, align 16
!CHECK:   %5 = alloca <2 x double>, i64 1, align 16
!CHECK:   %6 = load <2 x double>, ptr %2, align 16
!CHECK:   %7 = load <2 x double>, ptr %3, align 16
!CHECK:   %8 = load <2 x double>, ptr %4, align 16
!CHECK:   %9 = load <2 x double>, ptr %5, align 16
!CHECK:   %10 = bitcast <2 x double> %6 to <16 x i8>
!CHECK:   %11 = bitcast <2 x double> %7 to <16 x i8>
!CHECK:   %12 = bitcast <2 x double> %8 to <16 x i8>
!CHECK:   %13 = bitcast <2 x double> %9 to <16 x i8>
!CHECK:   %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:   store <512 x i1> %14, ptr %1, align 64

! mma_assemble_pair

      subroutine test_mma_assemble_pair_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i1

!CHECK: @test_mma_assemble_pair_i1_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <16 x i8>, ptr %1, align 16
!CHECK:  %5 = load <16 x i8>, ptr %2, align 16
!CHECK:  %6 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <256 x i1> %6, ptr %3, align 32

      subroutine test_mma_assemble_pair_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i2

!CHECK: @test_mma_assemble_pair_i2_
!CHECK:  %1 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <8 x i16>, ptr %1, align 16
!CHECK:  %5 = load <8 x i16>, ptr %2, align 16
!CHECK:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i4

!CHECK: @test_mma_assemble_pair_i4_
!CHECK:  %1 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <4 x i32>, ptr %1, align 16
!CHECK:  %5 = load <4 x i32>, ptr %2, align 16
!CHECK:  %6 = bitcast <4 x i32> %4 to <16 x i8>
!CHECK:  %7 = bitcast <4 x i32> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i8

!CHECK: @test_mma_assemble_pair_i8_
!CHECK:  %1 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <2 x i64>, ptr %1, align 16
!CHECK:  %5 = load <2 x i64>, ptr %2, align 16
!CHECK:  %6 = bitcast <2 x i64> %4 to <16 x i8>
!CHECK:  %7 = bitcast <2 x i64> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u1

!CHECK: @test_mma_assemble_pair_u1_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <16 x i8>, ptr %1, align 16
!CHECK:  %5 = load <16 x i8>, ptr %2, align 16
!CHECK:  %6 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <256 x i1> %6, ptr %3, align 32

      subroutine test_mma_assemble_pair_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u2

!CHECK: @test_mma_assemble_pair_u2_
!CHECK:  %1 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <8 x i16>, ptr %1, align 16
!CHECK:  %5 = load <8 x i16>, ptr %2, align 16
!CHECK:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u4

!CHECK: @test_mma_assemble_pair_u4_
!CHECK:  %1 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <4 x i32>, ptr %1, align 16
!CHECK:  %5 = load <4 x i32>, ptr %2, align 16
!CHECK:  %6 = bitcast <4 x i32> %4 to <16 x i8>
!CHECK:  %7 = bitcast <4 x i32> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u8

!CHECK: @test_mma_assemble_pair_u8_
!CHECK:  %1 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <2 x i64>, ptr %1, align 16
!CHECK:  %5 = load <2 x i64>, ptr %2, align 16
!CHECK:  %6 = bitcast <2 x i64> %4 to <16 x i8>
!CHECK:  %7 = bitcast <2 x i64> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_r4

!CHECK: @test_mma_assemble_pair_r4_
!CHECK:  %1 = alloca <4 x float>, i64 1, align 16
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <4 x float>, ptr %1, align 16
!CHECK:  %5 = load <4 x float>, ptr %2, align 16
!CHECK:  %6 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %7 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

      subroutine test_mma_assemble_pair_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_r8

!CHECK: @test_mma_assemble_pair_r8_
!CHECK:  %1 = alloca <2 x double>, i64 1, align 16
!CHECK:  %2 = alloca <2 x double>, i64 1, align 16
!CHECK:  %3 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %4 = load <2 x double>, ptr %1, align 16
!CHECK:  %5 = load <2 x double>, ptr %2, align 16
!CHECK:  %6 = bitcast <2 x double> %4 to <16 x i8>
!CHECK:  %7 = bitcast <2 x double> %5 to <16 x i8>
!CHECK:  %8 = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <256 x i1> %8, ptr %3, align 32

! mma_disassemble_acc

      subroutine test_mma_build_acc_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i1

!CHECK-LABEL: @test_mma_build_acc_i1
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %5 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %6 = load <16 x i8>, ptr %2, align 16
!CHECK:  %7 = load <16 x i8>, ptr %3, align 16
!CHECK:  %8 = load <16 x i8>, ptr %4, align 16
!CHECK:  %9 = load <16 x i8>, ptr %5, align 16
!CHECK:  %10 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %9, <16 x i8> %8, <16 x i8> %7, <16 x i8> %6)
!CHECK:  store <512 x i1> %10, ptr %1, align 64

      subroutine test_mma_build_acc_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i2

!CHECK-LABEL: @test_mma_build_acc_i2
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %4 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %5 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %6 = load <8 x i16>, ptr %2, align 16
!CHECK:  %7 = load <8 x i16>, ptr %3, align 16
!CHECK:  %8 = load <8 x i16>, ptr %4, align 16
!CHECK:  %9 = load <8 x i16>, ptr %5, align 16
!CHECK:  %10 = bitcast <8 x i16> %9 to <16 x i8>
!CHECK:  %11 = bitcast <8 x i16> %8 to <16 x i8>
!CHECK:  %12 = bitcast <8 x i16> %7 to <16 x i8>
!CHECK:  %13 = bitcast <8 x i16> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_mma_build_acc_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i4

!CHECK-LABEL: @test_mma_build_acc_i4
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %3 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %4 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %5 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %6 = load <4 x i32>, ptr %2, align 16
!CHECK:  %7 = load <4 x i32>, ptr %3, align 16
!CHECK:  %8 = load <4 x i32>, ptr %4, align 16
!CHECK:  %9 = load <4 x i32>, ptr %5, align 16
!CHECK:  %10 = bitcast <4 x i32> %9 to <16 x i8>
!CHECK:  %11 = bitcast <4 x i32> %8 to <16 x i8>
!CHECK:  %12 = bitcast <4 x i32> %7 to <16 x i8>
!CHECK:  %13 = bitcast <4 x i32> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_mma_build_acc_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i8

!CHECK-LABEL: @test_mma_build_acc_i8
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %3 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %4 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %5 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %6 = load <2 x i64>, ptr %2, align 16
!CHECK:  %7 = load <2 x i64>, ptr %3, align 16
!CHECK:  %8 = load <2 x i64>, ptr %4, align 16
!CHECK:  %9 = load <2 x i64>, ptr %5, align 16
!CHECK:  %10 = bitcast <2 x i64> %9 to <16 x i8>
!CHECK:  %11 = bitcast <2 x i64> %8 to <16 x i8>
!CHECK:  %12 = bitcast <2 x i64> %7 to <16 x i8>
!CHECK:  %13 = bitcast <2 x i64> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_mma_build_acc_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u1

!CHECK-LABEL: @test_mma_build_acc_u1
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %5 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %6 = load <16 x i8>, ptr %2, align 16
!CHECK:  %7 = load <16 x i8>, ptr %3, align 16
!CHECK:  %8 = load <16 x i8>, ptr %4, align 16
!CHECK:  %9 = load <16 x i8>, ptr %5, align 16
!CHECK:  %10 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %9, <16 x i8> %8, <16 x i8> %7, <16 x i8> %6)
!CHECK:  store <512 x i1> %10, ptr %1, align 64

      subroutine test_mma_build_acc_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u2

!CHECK-LABEL: @test_mma_build_acc_u2
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %4 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %5 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %6 = load <8 x i16>, ptr %2, align 16
!CHECK:  %7 = load <8 x i16>, ptr %3, align 16
!CHECK:  %8 = load <8 x i16>, ptr %4, align 16
!CHECK:  %9 = load <8 x i16>, ptr %5, align 16
!CHECK:  %10 = bitcast <8 x i16> %9 to <16 x i8>
!CHECK:  %11 = bitcast <8 x i16> %8 to <16 x i8>
!CHECK:  %12 = bitcast <8 x i16> %7 to <16 x i8>
!CHECK:  %13 = bitcast <8 x i16> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_mma_build_acc_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u4

!CHECK-LABEL: @test_mma_build_acc_u4
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %3 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %4 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %5 = alloca <4 x i32>, i64 1, align 16
!CHECK:  %6 = load <4 x i32>, ptr %2, align 16
!CHECK:  %7 = load <4 x i32>, ptr %3, align 16
!CHECK:  %8 = load <4 x i32>, ptr %4, align 16
!CHECK:  %9 = load <4 x i32>, ptr %5, align 16
!CHECK:  %10 = bitcast <4 x i32> %9 to <16 x i8>
!CHECK:  %11 = bitcast <4 x i32> %8 to <16 x i8>
!CHECK:  %12 = bitcast <4 x i32> %7 to <16 x i8>
!CHECK:  %13 = bitcast <4 x i32> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

      subroutine test_mma_build_acc_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u8

!CHECK-LABEL: @test_mma_build_acc_u8
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %3 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %4 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %5 = alloca <2 x i64>, i64 1, align 16
!CHECK:  %6 = load <2 x i64>, ptr %2, align 16
!CHECK:  %7 = load <2 x i64>, ptr %3, align 16
!CHECK:  %8 = load <2 x i64>, ptr %4, align 16
!CHECK:  %9 = load <2 x i64>, ptr %5, align 16
!CHECK:  %10 = bitcast <2 x i64> %9 to <16 x i8>
!CHECK:  %11 = bitcast <2 x i64> %8 to <16 x i8>
!CHECK:  %12 = bitcast <2 x i64> %7 to <16 x i8>
!CHECK:  %13 = bitcast <2 x i64> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64


      subroutine test_mma_build_acc_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_r4

!CHECK-LABEL: @test_mma_build_acc_r4
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = alloca <4 x float>, i64 1, align 16
!CHECK:  %5 = alloca <4 x float>, i64 1, align 16
!CHECK:  %6 = load <4 x float>, ptr %2, align 16
!CHECK:  %7 = load <4 x float>, ptr %3, align 16
!CHECK:  %8 = load <4 x float>, ptr %4, align 16
!CHECK:  %9 = load <4 x float>, ptr %5, align 16
!CHECK:  %10 = bitcast <4 x float> %9 to <16 x i8>
!CHECK:  %11 = bitcast <4 x float> %8 to <16 x i8>
!CHECK:  %12 = bitcast <4 x float> %7 to <16 x i8>
!CHECK:  %13 = bitcast <4 x float> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64


      subroutine test_mma_build_acc_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_r8

!CHECK-LABEL: @test_mma_build_acc_r8
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <2 x double>, i64 1, align 16
!CHECK:  %3 = alloca <2 x double>, i64 1, align 16
!CHECK:  %4 = alloca <2 x double>, i64 1, align 16
!CHECK:  %5 = alloca <2 x double>, i64 1, align 16
!CHECK:  %6 = load <2 x double>, ptr %2, align 16
!CHECK:  %7 = load <2 x double>, ptr %3, align 16
!CHECK:  %8 = load <2 x double>, ptr %4, align 16
!CHECK:  %9 = load <2 x double>, ptr %5, align 16
!CHECK:  %10 = bitcast <2 x double> %9 to <16 x i8>
!CHECK:  %11 = bitcast <2 x double> %8 to <16 x i8>
!CHECK:  %12 = bitcast <2 x double> %7 to <16 x i8>
!CHECK:  %13 = bitcast <2 x double> %6 to <16 x i8>
!CHECK:  %14 = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %10, <16 x i8> %11, <16 x i8> %12, <16 x i8> %13)
!CHECK:  store <512 x i1> %14, ptr %1, align 64

! mma_disassemble_acc

      subroutine test_disassemble_acc()
      use, intrinsic :: mma
      implicit none
      __vector_quad :: vq
      real :: data
      call mma_disassemble_acc(data, vq)
      end subroutine

!CHECK-LABEL: @test_disassemble_acc_
!CHECK:  %1 = alloca float, i64 1, align 4
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = load <512 x i1>, ptr %2, align 64
!CHECK:  %4 = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1> %3)
!CHECK:  store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %4, ptr %1, align 16

! mma_disassemble_pair

      subroutine test_disassemble_pair()
      use, intrinsic :: mma
      implicit none
      __vector_pair :: vp
      real :: data
      call mma_disassemble_pair(data, vp)
      end subroutine

!CHECK-LABEL: @test_disassemble_pair_
!CHECK:  %1 = alloca float, i64 1, align 4
!CHECK:  %2 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %3 = load <256 x i1>, ptr %2, align 32
!CHECK:  %4 = call { <16 x i8>, <16 x i8> } @llvm.ppc.vsx.disassemble.pair(<256 x i1> %3)
!CHECK:  store { <16 x i8>, <16 x i8> } %4, ptr %1, align 16
