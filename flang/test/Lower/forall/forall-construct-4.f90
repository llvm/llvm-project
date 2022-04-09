! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** Test forall targeted by label
subroutine test4_forall_construct()
      integer :: a(2) = 1
100   forall (i=1:2)
        a(i) = a(i) + 1
      end forall
      if (a(1) > 3) goto 200
      goto 100
200   return
end subroutine test4_forall_construct

! CHECK-LABEL: func @_QPtest4_forall_construct
! CHECK:   cf.br ^bb1
! CHECK: ^bb1:  // 2 preds: ^bb0, ^bb2
! CHECK:   %{{.*}} = fir.do_loop
! CHECK:   cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK: ^bb2:  // pred: ^bb1
! CHECK:   cf.br ^bb1
! CHECK: ^bb3:  // pred: ^bb1
! CHECK:   cf.br ^bb4
! CHECK: ^bb4:  // pred: ^bb3
! CHECK:   return

subroutine test4_forall_construct2()
      integer :: a(2) = 1
100   forall (i=1:2) a(i) = a(i) + 1
      if (a(1) > 3) goto 200
      goto 100
200   return
end subroutine test4_forall_construct2

! CHECK-LABEL: func @_QPtest4_forall_construct2
! CHECK:   cf.br ^bb1
! CHECK: ^bb1:  // 2 preds: ^bb0, ^bb2
! CHECK:   %{{.*}} = fir.do_loop
! CHECK:   cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK: ^bb2:  // pred: ^bb1
! CHECK:   cf.br ^bb1
! CHECK: ^bb3:  // pred: ^bb1
! CHECK:   cf.br ^bb4
! CHECK: ^bb4:  // pred: ^bb3
! CHECK:   return
