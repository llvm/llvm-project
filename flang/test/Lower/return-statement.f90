! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

program basic
  return
end program

! CHECK-LABEL: func @_QQmain() {
! CHECK:         return
! CHECK:       }

subroutine sub1()
  return
end

! CHECK-LABEL: func @_QPsub1() {
! CHECK:         cf.br ^bb1
! CHECK:       ^bb1:  // pred: ^bb0
! CHECK:         return

subroutine sub2()
  goto 3
  2 return
  3 goto 2
end

! CHECK-LABEL: func @_QPsub2() {
! CHECK:         cf.br ^bb2
! CHECK:       ^bb1:  // pred: ^bb2
! CHECK:         cf.br ^bb3
! CHECK:       ^bb2:  // pred: ^bb0
! CHECK:         cf.br ^bb1
! CHECK:       ^bb3:  // pred: ^bb1
! CHECK:         return

