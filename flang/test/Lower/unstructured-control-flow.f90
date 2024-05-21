!RUN: bbc -emit-hlfir -o - %s | FileCheck %s

!CHECK-LABEL: func.func @_QPunstructured1
!CHECK:   fir.select %{{[0-9]+}} : i32 [{{.*}}, ^bb[[BLOCK3:[0-9]+]], {{.*}}, ^bb[[BLOCK4:[0-9]+]], {{.*}}, ^bb[[BLOCK5:[0-9]+]], {{.*}}, ^bb[[BLOCK1:[0-9]+]]]
!CHECK: ^bb[[BLOCK1]]:
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[BLOCK2:[0-9]+]], ^bb[[BLOCK4]]
!CHECK: ^bb[[BLOCK2]]:
!CHECK:   fir.if
!CHECK:   cf.br ^bb[[BLOCK3]]
!CHECK: ^bb[[BLOCK3]]:
!CHECK:   %[[C10:[a-z0-9_]+]] = arith.constant 10 : i32
!CHECK:   arith.addi {{.*}}, %[[C10]]
!CHECK:   cf.br ^bb[[BLOCK4]]
!CHECK: ^bb[[BLOCK4]]:
!CHECK:   %[[C100:[a-z0-9_]+]] = arith.constant 100 : i32
!CHECK:   arith.addi {{.*}}, %[[C100]]
!CHECK:   cf.br ^bb[[BLOCK5]]
!CHECK: ^bb[[BLOCK5]]:
!CHECK:   %[[C1000:[a-z0-9_]+]] = arith.constant 1000 : i32
!CHECK:   arith.addi {{.*}}, %[[C1000]]
!CHECK:   return
subroutine unstructured1(j, k)
    goto (11, 22, 33) j-3  ! computed goto - an expression outside [1,3] is a nop
    if (j == 2) goto 22
    if (j == 1) goto 11
    k = k + 1
11  k = k + 10
22  k = k + 100
33  k = k + 1000
end

