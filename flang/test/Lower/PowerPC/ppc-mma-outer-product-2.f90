! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

      subroutine test_xvbf16ger2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2(cq, vu10, vu11)
      end subroutine test_xvbf16ger2

!CHECK-LABEL: @test_xvbf16ger2_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = call <512 x i1> @llvm.ppc.mma.xvbf16ger2(<16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %6, ptr %1, align 64


      subroutine test_xvbf16ger2nn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2nn(cq, vu10, vu11)
      end subroutine test_xvbf16ger2nn

!CHECK-LABEL: @test_xvbf16ger2nn_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvbf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvbf16ger2np()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2np(cq, vu10, vu11)
      end subroutine test_xvbf16ger2np

!CHECK-LABEL: @test_xvbf16ger2np_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvbf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvbf16ger2pn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2pn(cq, vu10, vu11)
      end subroutine test_xvbf16ger2pn

!CHECK-LABEL: @test_xvbf16ger2pn_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvbf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvbf16ger2pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2pp(cq, vu10, vu11)
      end subroutine test_xvbf16ger2pp

!CHECK-LABEL: @test_xvbf16ger2pp_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvbf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf16ger2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2(cq, vu10, vu11)
      end subroutine test_xvf16ger2

!CHECK-LABEL: @test_xvf16ger2_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvf16ger2(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_xvf16ger2nn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2nn(cq, vu10, vu11)
      end subroutine test_xvf16ger2nn

!CHECK-LABEL: @test_xvf16ger2nn_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf16ger2nn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf16ger2np()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2np(cq, vu10, vu11)
      end subroutine test_xvf16ger2np

!CHECK-LABEL: @test_xvf16ger2np_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf16ger2np(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf16ger2pn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2pn(cq, vu10, vu11)
      end subroutine test_xvf16ger2pn

!CHECK-LABEL: @test_xvf16ger2pn_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf16ger2pn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf16ger2pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2pp(cq, vu10, vu11)
      end subroutine test_xvf16ger2pp

!CHECK-LABEL: @test_xvf16ger2pp_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf32ger_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32ger(cq, vu10, vu11)
      end subroutine test_xvf32ger_u1

!CHECK-LABEL: @test_xvf32ger_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvf32ger(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %1, align 64


      subroutine test_xvf32ger_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32ger(cq, vr40, vr41)
      end subroutine test_xvf32ger_r4

!CHECK-LABEL: @test_xvf32ger_r4_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = load <4 x float>, ptr %2, align 16
!CHECK:  %5 = load <4 x float>, ptr %3, align 16
!CHECK:  %6 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %7 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %8 = call <512 x i1> @llvm.ppc.mma.xvf32ger(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_xvf32gernn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gernn(cq, vu10, vu11)
      end subroutine test_xvf32gernn_u1

!CHECK-LABEL: @test_xvf32gernn_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf32gernn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf32gernn_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gernn(cq, vr40, vr41)
      end subroutine test_xvf32gernn_r4

!CHECK-LABEL: @test_xvf32gernn_r4_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = load <4 x float>, ptr %2, align 16
!CHECK:  %5 = load <4 x float>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %8 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %9 = call <512 x i1> @llvm.ppc.mma.xvf32gernn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvf32gernp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gernp(cq, vu10, vu11)
      end subroutine test_xvf32gernp_u1

!CHECK-LABEL: @test_xvf32gernp_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf32gernp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf32gernp_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gernp(cq, vr40, vr41)
      end subroutine test_xvf32gernp_r4

!CHECK-LABEL: @test_xvf32gernp_r4_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = load <4 x float>, ptr %2, align 16
!CHECK:  %5 = load <4 x float>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %8 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %9 = call <512 x i1> @llvm.ppc.mma.xvf32gernp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvf32gerpn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gerpn(cq, vu10, vu11)
      end subroutine test_xvf32gerpn_u1

!CHECK-LABEL: @test_xvf32gerpn_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf32gerpn(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvf32gerpn_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gerpn(cq, vr40, vr41)
      end subroutine test_xvf32gerpn_r4

!CHECK-LABEL: @test_xvf32gerpn_r4_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = load <4 x float>, ptr %2, align 16
!CHECK:  %5 = load <4 x float>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %8 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %9 = call <512 x i1> @llvm.ppc.mma.xvf32gerpn(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvf32gerpp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gerpp(cq, vu10, vu11)
      end subroutine test_xvf32gerpp_u1

!CHECK-LABEL: @test_xvf32gerpp_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf32gerpp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64


      subroutine test_xvf32gerpp_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gerpp(cq, vr40, vr41)
      end subroutine test_xvf32gerpp_r4

!CHECK-LABEL: @test_xvf32gerpp_r4_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <4 x float>, i64 1, align 16
!CHECK:  %3 = alloca <4 x float>, i64 1, align 16
!CHECK:  %4 = load <4 x float>, ptr %2, align 16
!CHECK:  %5 = load <4 x float>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = bitcast <4 x float> %4 to <16 x i8>
!CHECK:  %8 = bitcast <4 x float> %5 to <16 x i8>
!CHECK:  %9 = call <512 x i1> @llvm.ppc.mma.xvf32gerpp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:  store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvf64ger_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64ger(cq, cp, vu10)
      end subroutine test_xvf64ger_u1

!CHECK-LABEL: @test_xvf64ger_u1_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvf64ger(<256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %2, align 64

      subroutine test_xvf64ger_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64ger(cq, cp, vr80)
      end subroutine test_xvf64ger_r8

!CHECK-LABEL: @test_xvf64ger_r8_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <2 x double>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <2 x double>, ptr %3, align 16
!CHECK:  %6 = bitcast <2 x double> %5 to <16 x i8>
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64ger(<256 x i1> %4, <16 x i8> %6)
!CHECK:  store <512 x i1> %7, ptr %2, align 64


      subroutine test_xvf64gernn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernn(cq, cp, vu10)
      end subroutine test_xvf64gernn_u1

!CHECK-LABEL: @test_xvf64gernn_u1_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %2, align 64


      subroutine test_xvf64gernn_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernn(cq, cp, vr80)
      end subroutine test_xvf64gernn_r8

!CHECK-LABEL: @test_xvf64gernn_r8_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <2 x double>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <2 x double>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = bitcast <2 x double> %5 to <16 x i8>
!CHECK:  %8 = call <512 x i1> @llvm.ppc.mma.xvf64gernn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7)
!CHECK:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_xvf64gernp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernp(cq, cp, vu10)
      end subroutine test_xvf64gernp_u1

!CHECK-LABEL: @test_xvf64gernp_u1_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_xvf64gernp_r8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernp(cq, cp, vr80)
      end subroutine test_xvf64gernp_r8

!CHECK-LABEL: @test_xvf64gernp_r8_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64gernp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_xvf64gerpn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpn(cq, cp, vu10)
      end subroutine test_xvf64gerpn_u1

!CHECK-LABEL: @test_xvf64gerpn_u1_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %2, align 64

      subroutine test_xvf64gerpn_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpn(cq, cp, vr80)
      end subroutine test_xvf64gerpn_r8

!CHECK-LABEL: @test_xvf64gerpn_r8_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <2 x double>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <2 x double>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = bitcast <2 x double> %5 to <16 x i8>
!CHECK:  %8 = call <512 x i1> @llvm.ppc.mma.xvf64gerpn(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7)
!CHECK:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_xvf64gerpp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpp(cq, cp, vu10)
      end subroutine test_xvf64gerpp_u1

!CHECK-LABEL: @test_xvf64gerpp_u1_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %2, align 64


      subroutine test_xvf64gerpp_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpp(cq, cp, vr80)
      end subroutine test_xvf64gerpp_r8

!CHECK-LABEL: @test_xvf64gerpp_r8_
!CHECK:  %1 = alloca <256 x i1>, i64 1, align 32
!CHECK:  %2 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %3 = alloca <2 x double>, i64 1, align 16
!CHECK:  %4 = load <256 x i1>, ptr %1, align 32
!CHECK:  %5 = load <2 x double>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %2, align 64
!CHECK:  %7 = bitcast <2 x double> %5 to <16 x i8>
!CHECK:  %8 = call <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1> %6, <256 x i1> %4, <16 x i8> %7)
!CHECK:  store <512 x i1> %8, ptr %2, align 64

      subroutine test_xvi16ger2_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2(cq, vu10, vu11)
      end subroutine test_xvi16ger2_u1

!CHECK-LABEL: @test_xvi16ger2_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvi16ger2(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_xvi16ger2_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2(cq, vi20, vi21)
      end subroutine test_xvi16ger2_i2

!CHECK-LABEL: @test_xvi16ger2_i2_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:  %4 = load <8 x i16>, ptr %2, align 16
!CHECK:  %5 = load <8 x i16>, ptr %3, align 16
!CHECK:  %6 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:  %7 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:  %8 = call <512 x i1> @llvm.ppc.mma.xvi16ger2(<16 x i8> %6, <16 x i8> %7)
!CHECK:  store <512 x i1> %8, ptr %1, align 64

      subroutine test_xvi16ger2pp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2pp(cq, vu10, vu11)
      end subroutine test_xvi16ger2pp_u1

!CHECK-LABEL: @test_xvi16ger2pp_u1_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = call <512 x i1> @llvm.ppc.mma.xvi16ger2pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi16ger2pp_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2pp(cq, vi20, vi21)
      end subroutine test_xvi16ger2pp_i2

!CHECK-LABEL: @test_xvi16ger2pp_i2_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %4 = load <8 x i16>, ptr %2, align 16
!CHECK:   %5 = load <8 x i16>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:   %8 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:   %9 = call <512 x i1> @llvm.ppc.mma.xvi16ger2pp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:   store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvi16ger2s_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2s(cq, vu10, vu11)
      end subroutine test_xvi16ger2s_u1

!CHECK-LABEL:  @test_xvi16ger2s_u1_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = call <512 x i1> @llvm.ppc.mma.xvi16ger2s(<16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %6, ptr %1, align 64

      subroutine test_xvi16ger2s_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2s(cq, vi20, vi21)
      end subroutine test_xvi16ger2s_i2

!CHECK-LABEL:  @test_xvi16ger2s_i2_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %4 = load <8 x i16>, ptr %2, align 16
!CHECK:   %5 = load <8 x i16>, ptr %3, align 16
!CHECK:   %6 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:   %7 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:   %8 = call <512 x i1> @llvm.ppc.mma.xvi16ger2s(<16 x i8> %6, <16 x i8> %7)
!CHECK:   store <512 x i1> %8, ptr %1, align 64

      subroutine test_xvi16ger2spp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2spp(cq, vu10, vu11)
      end subroutine test_xvi16ger2spp_u1

!CHECK-LABEL:  @test_xvi16ger2spp_u1_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = call <512 x i1> @llvm.ppc.mma.xvi16ger2spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi16ger2spp_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2spp(cq, vi20, vi21)
      end subroutine test_xvi16ger2spp_i2

!CHECK-LABEL:  @test_xvi16ger2spp_i2_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %3 = alloca <8 x i16>, i64 1, align 16
!CHECK:   %4 = load <8 x i16>, ptr %2, align 16
!CHECK:   %5 = load <8 x i16>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = bitcast <8 x i16> %4 to <16 x i8>
!CHECK:   %8 = bitcast <8 x i16> %5 to <16 x i8>
!CHECK:   %9 = call <512 x i1> @llvm.ppc.mma.xvi16ger2spp(<512 x i1> %6, <16 x i8> %7, <16 x i8> %8)
!CHECK:   store <512 x i1> %9, ptr %1, align 64

      subroutine test_xvi4ger8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi4ger8(cq, vu10, vu11)
      end subroutine test_xvi4ger8

!CHECK-LABEL:  @test_xvi4ger8_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = call <512 x i1> @llvm.ppc.mma.xvi4ger8(<16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %6, ptr %1, align 64

      subroutine test_xvi4ger8pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi4ger8pp(cq, vu10, vu11)
      end subroutine test_xvi4ger8pp

!CHECK-LABEL:  @test_xvi4ger8pp_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = call <512 x i1> @llvm.ppc.mma.xvi4ger8pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi8ger4_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4(cq, vu10, vu11)
      end subroutine test_xvi8ger4_u1

!CHECK-LABEL: @test_xvi8ger4_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvi8ger4(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %1, align 64


      subroutine test_xvi8ger4_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4(cq, vi10, vi11)
      end subroutine test_xvi8ger4_i1

!CHECK-LABEL: @test_xvi8ger4_i1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = call <512 x i1> @llvm.ppc.mma.xvi8ger4(<16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %6, ptr %1, align 64

      subroutine test_xvi8ger4pp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4pp(cq, vu10, vu11)
      end subroutine test_xvi8ger4pp_u1

!CHECK-LABEL: @test_xvi8ger4pp_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi8ger4pp_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4pp(cq, vi10, vi11)
      end subroutine test_xvi8ger4pp_i1

!CHECK-LABEL:  @test_xvi8ger4pp_i1_
!CHECK:   %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:   %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:   %4 = load <16 x i8>, ptr %2, align 16
!CHECK:   %5 = load <16 x i8>, ptr %3, align 16
!CHECK:   %6 = load <512 x i1>, ptr %1, align 64
!CHECK:   %7 = call <512 x i1> @llvm.ppc.mma.xvi8ger4pp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:   store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi8ger4spp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4spp(cq, vu10, vu11)
      end subroutine test_xvi8ger4spp_u1

!CHECK-LABEL: @test_xvi8ger4spp_u1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64

      subroutine test_xvi8ger4spp_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4spp(cq, vi10, vi11)
      end subroutine test_xvi8ger4spp_i1

!CHECK-LABEL: @test_xvi8ger4spp_i1_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %4 = load <16 x i8>, ptr %2, align 16
!CHECK:  %5 = load <16 x i8>, ptr %3, align 16
!CHECK:  %6 = load <512 x i1>, ptr %1, align 64
!CHECK:  %7 = call <512 x i1> @llvm.ppc.mma.xvi8ger4spp(<512 x i1> %6, <16 x i8> %4, <16 x i8> %5)
!CHECK:  store <512 x i1> %7, ptr %1, align 64
