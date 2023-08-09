! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}
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
