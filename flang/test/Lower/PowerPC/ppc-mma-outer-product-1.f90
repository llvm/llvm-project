! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

      subroutine test_pmxvbf16ger2_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2_def

!CHECK-LABEL: @test_pmxvbf16ger2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64


      subroutine test_pmxvbf16ger2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2_non_def

!CHECK-LABEL: @test_pmxvbf16ger2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64


      subroutine test_pmxvbf16ger2nn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2nn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2nn_def

!CHECK-LABEL: @test_pmxvbf16ger2nn_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2nn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2nn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2nn_non_def

!CHECK-LABEL: @test_pmxvbf16ger2nn_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2np_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2np(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2np_def

!CHECK-LABEL: @test_pmxvbf16ger2np_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2np_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2np(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2np_non_def

!CHECK-LABEL: @test_pmxvbf16ger2np_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2pn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2pn_def

!CHECK-LABEL: @test_pmxvbf16ger2pn_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2pn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2pn_non_def

!CHECK-LABEL: @test_pmxvbf16ger2pn_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2pp_def

!CHECK-LABEL: @test_pmxvbf16ger2pp_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvbf16ger2pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2pp_non_def

!CHECK-LABEL: @test_pmxvbf16ger2pp_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2_def

!CHECK-LABEL: @test_pmxvf16ger2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvf16ger2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2_non_def

!CHECK-LABEL: @test_pmxvf16ger2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvf16ger2nn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2nn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2nn_def

!CHECK-LABEL: @test_pmxvf16ger2nn_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2nn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2nn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2nn_non_def

!CHECK-LABEL: @test_pmxvf16ger2nn_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2np_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2np(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2np_def

!CHECK-LABEL: @test_pmxvf16ger2np_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2np_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2np(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2np_non_def

!CHECK-LABEL: @test_pmxvf16ger2np_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2pn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2pn_def

!CHECK-LABEL: @test_pmxvf16ger2pn_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2pn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2pn_non_def

!CHECK-LABEL: @test_pmxvf16ger2pn_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2pp_def

!CHECK-LABEL: @test_pmxvf16ger2pp_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf16ger2pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2pp_non_def

!CHECK-LABEL: @test_pmxvf16ger2pp_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32ger_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32ger_u1_def

!CHECK-LABEL: @test_pmxvf32ger_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvf32ger_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32ger_u1_non_def

!CHECK-LABEL: @test_pmxvf32ger_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvf32ger_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32ger_r4_def

!CHECK-LABEL: @test_pmxvf32ger_r4_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %6, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvf32ger_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32ger_r4_non_def

!CHECK-LABEL: @test_pmxvf32ger_r4_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %6, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvf32gernn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gernn_u1_def

!CHECK-LABEL: @test_pmxvf32gernn_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gernn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gernn_u1_non_def

!CHECK-LABEL: @test_pmxvf32gernn_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gernn_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gernn_r4_def

!CHECK-LABEL: @test_pmxvf32gernn_r4_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gernn_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gernn_r4_non_def

!CHECK-LABEL: @test_pmxvf32gernn_r4_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gernp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gernp_u1_def

!CHECK-LABEL: @test_pmxvf32gernp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gernp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gernp_u1_non_def

!CHECK-LABEL: @test_pmxvf32gernp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gernp_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gernp_r4_def

!CHECK-LABEL: @test_pmxvf32gernp_r4_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gernp_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gernp_r4_non_def

!CHECK-LABEL: @test_pmxvf32gernp_r4_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gerpn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gerpn_u1_def

!CHECK-LABEL: @test_pmxvf32gerpn_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gerpn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gerpn_u1_non_def

!CHECK-LABEL: @test_pmxvf32gerpn_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gerpn_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gerpn_r4_def

!CHECK-LABEL: @test_pmxvf32gerpn_r4_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gerpn_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gerpn_r4_non_def

!CHECK-LABEL: @test_pmxvf32gerpn_r4_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gerpp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gerpp_u1_def

!CHECK-LABEL: @test_pmxvf32gerpp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gerpp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gerpp_u1_non_def

!CHECK-LABEL: @test_pmxvf32gerpp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvf32gerpp_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gerpp_r4_def

!CHECK-LABEL: @test_pmxvf32gerpp_r4_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf32gerpp_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gerpp_r4_non_def

!CHECK-LABEL: @test_pmxvf32gerpp_r4_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %3 = alloca <4 x float>, i64 1, align 16
!LLVMIR:  %4 = load <4 x float>, ptr %2, align 16
!LLVMIR:  %5 = load <4 x float>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <4 x float> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <4 x float> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvf64ger_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64ger_u1_def

!CHECK-LABEL: @test_pmxvf64ger_u1_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %2, align 64

      subroutine test_pmxvf64ger_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64ger_u1_non_def

!CHECK-LABEL: @test_pmxvf64ger_u1_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %2, align 64

      subroutine test_pmxvf64ger_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64ger_r8_def

!CHECK-LABEL: @test_pmxvf64ger_r8_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %4, <16 x i8> %6, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64ger_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64ger_r8_non_def

!CHECK-LABEL: @test_pmxvf64ger_r8_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %4, <16 x i8> %6, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gernn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gernn_u1_def

!CHECK-LABEL: @test_pmxvf64gernn_u1_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gernn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gernn_u1_non_def

!CHECK-LABEL: @test_pmxvf64gernn_u1_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gernn_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gernn_r8_def

!CHECK-LABEL: @test_pmxvf64gernn_r8_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gernn_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gernn_r8_non_def

!CHECK-LABEL: @test_pmxvf64gernn_r8_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gernp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gernp_u1_def

!CHECK-LABEL: @test_pmxvf64gernp_u1_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gernp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gernp_u1_non_def

!CHECK-LABEL: @test_pmxvf64gernp_u1_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gernp_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gernp_r8_def

!CHECK-LABEL: @test_pmxvf64gernp_r8_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gernp_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gernp_r8_non_def

!CHECK-LABEL: @test_pmxvf64gernp_r8_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gerpn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gerpn_u1_def

!CHECK-LABEL: @test_pmxvf64gerpn_u1_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gerpn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gerpn_u1_non_def

!CHECK-LABEL: @test_pmxvf64gerpn_u1_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gerpn_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gerpn_r8_def

!CHECK-LABEL: @test_pmxvf64gerpn_r8_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gerpn_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gerpn_r8_non_def

!CHECK-LABEL: @test_pmxvf64gerpn_r8_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gerpp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gerpp_u1_def

!CHECK-LABEL: @test_pmxvf64gerpp_u1_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gerpp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gerpp_u1_non_def

!CHECK-LABEL: @test_pmxvf64gerpp_u1_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_pmxvf64gerpp_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gerpp_r8_def

!CHECK-LABEL: @test_pmxvf64gerpp_r8_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvf64gerpp_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gerpp_r8_non_def

!CHECK-LABEL: @test_pmxvf64gerpp_r8_non_def_
!LLVMIR:  %1 = alloca <256 x i1>, i64 1, align 32
!LLVMIR:  %2 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %3 = alloca <2 x double>, i64 1, align 16
!LLVMIR:  %4 = load <256 x i1>, ptr %1, align 32
!LLVMIR:  %5 = load <2 x double>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %2, align 64
!LLVMIR:  %7 = bitcast <2 x double> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_pmxvi16ger2_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2_u1_def

!CHECK-LABEL: @test_pmxvi16ger2_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi16ger2_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi16ger2_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2_i2_def

!CHECK-LABEL: @test_pmxvi16ger2_i2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %6, <16 x i8> %7, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvi16ger2_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2_i2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %6, <16 x i8> %7, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvi16ger2pp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2pp_u1_def

!CHECK-LABEL: @test_pmxvi16ger2pp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi16ger2pp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2pp_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2pp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi16ger2pp_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2pp_i2_def

!CHECK-LABEL: @test_pmxvi16ger2pp_i2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvi16ger2pp_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2pp_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2pp_i2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvi16ger2s_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2s_u1_def

!CHECK-LABEL: @test_pmxvi16ger2s_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi16ger2s_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2s_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2s_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi16ger2s_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2s_i2_def

!CHECK-LABEL: @test_pmxvi16ger2s_i2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %6, <16 x i8> %7, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvi16ger2s_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2s_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2s_i2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %8 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %6, <16 x i8> %7, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_pmxvi16ger2spp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2spp_u1_def

!CHECK-LABEL: @test_pmxvi16ger2spp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi16ger2spp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2spp_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2spp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi16ger2spp_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2spp_i2_def

!CHECK-LABEL: @test_pmxvi16ger2spp_i2_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_pmxvi16ger2spp_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2spp_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2spp_i2_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %3 = alloca <8 x i16>, i64 1, align 16
!LLVMIR:  %4 = load <8 x i16>, ptr %2, align 16
!LLVMIR:  %5 = load <8 x i16>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = bitcast <8 x i16> %4 to <16 x i8>
!LLVMIR:  %8 = bitcast <8 x i16> %5 to <16 x i8>
!LLVMIR:  %9 = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %9, ptr %1, align 64


      subroutine test_pmxvi4ger8_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi4ger8_def

!CHECK-LABEL: @test_pmxvi4ger8_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi4ger8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi4ger8_non_def

!CHECK-LABEL: @test_pmxvi4ger8_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi4ger8pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi4ger8pp_def

!CHECK-LABEL: @test_pmxvi4ger8pp_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi4ger8pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi4ger8pp_non_def

!CHECK-LABEL: @test_pmxvi4ger8pp_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4_u1_def

!CHECK-LABEL: @test_pmxvi8ger4_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi8ger4_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi8ger4_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4_i1_def

!CHECK-LABEL: @test_pmxvi8ger4_i1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi8ger4_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4_i1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_pmxvi8ger4pp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4pp_u1_def

!CHECK-LABEL: @test_pmxvi8ger4pp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4pp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4pp_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4pp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4pp_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4pp_i1_def

!CHECK-LABEL: @test_pmxvi8ger4pp_i1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4pp_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4pp_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4pp_i1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4spp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4spp_u1_def

!CHECK-LABEL: @test_pmxvi8ger4spp_u1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4spp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4spp_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4spp_u1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4spp_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4spp_i1_def

!CHECK-LABEL: @test_pmxvi8ger4spp_i1_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_pmxvi8ger4spp_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4spp_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4spp_i1_non_def_
!LLVMIR:  %1 = alloca <512 x i1>, i64 1, align 64
!LLVMIR:  %2 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %3 = alloca <16 x i8>, i64 1, align 16
!LLVMIR:  %4 = load <16 x i8>, ptr %2, align 16
!LLVMIR:  %5 = load <16 x i8>, ptr %3, align 16
!LLVMIR:  %6 = load <512 x i1>, ptr %1, align 64
!LLVMIR:  %7 = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5, i32 7, i32 7, i32 2)
!LLVMIR:  store <512 x i1> %7, ptr %1, align 64
