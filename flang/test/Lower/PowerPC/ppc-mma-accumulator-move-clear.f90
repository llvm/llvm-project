! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

      subroutine test_xxmfacc()
      use, intrinsic :: mma
      implicit none
      __vector_quad :: cq
      call mma_xxmfacc(cq)
      end subroutine test_xxmfacc

!CHECK-LABEL: @test_xxmfacc_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = load <512 x i1>, ptr %1, align 64
!CHECK:  %3 = call <512 x i1> @llvm.ppc.mma.xxmfacc(<512 x i1> %2)
!CHECK:  store <512 x i1> %3, ptr %1, align 64

      subroutine test_xxmtacc()
      use, intrinsic :: mma
      implicit none
      __vector_quad :: cq
      call mma_xxmtacc(cq)
      end subroutine test_xxmtacc

!CHECK-LABEL: @test_xxmtacc_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = load <512 x i1>, ptr %1, align 64
!CHECK:  %3 = call <512 x i1> @llvm.ppc.mma.xxmtacc(<512 x i1> %2)
!CHECK:  store <512 x i1> %3, ptr %1, align 64

      subroutine test_xxsetaccz()
      use, intrinsic :: mma
      implicit none
      __vector_quad :: cq
      call mma_xxsetaccz(cq)
      end subroutine test_xxsetaccz

!CHECK-LABEL: @test_xxsetaccz_
!CHECK:  %1 = alloca <512 x i1>, i64 1, align 64
!CHECK:  %2 = call <512 x i1> @llvm.ppc.mma.xxsetaccz()
!CHECK:  store <512 x i1> %2, ptr %1, align 64
