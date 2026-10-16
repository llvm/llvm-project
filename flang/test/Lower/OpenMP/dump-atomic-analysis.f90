!RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -emit-hlfir -mmlir -fdebug-dump-atomic-analysis %s -o /dev/null 2>&1 | FileCheck %s

subroutine f00(x)
  integer :: x, v
  !$omp atomic read
    v = x
end

!CHECK: Analysis {
!CHECK-NEXT:   atom: x
!CHECK-NEXT:   cond: <null>
!CHECK-NEXT:   op0 {
!CHECK-NEXT:     what: Read
!CHECK-NEXT:     assign: v=x
!CHECK-NEXT:   }
!CHECK-NEXT:   op1 {
!CHECK-NEXT:     what: None
!CHECK-NEXT:     assign: <null>
!CHECK-NEXT:   }
!CHECK-NEXT: }


subroutine f01(v)
  integer :: x, v
  !$omp atomic write
    x = v
end

!CHECK: Analysis {
!CHECK-NEXT:   atom: x
!CHECK-NEXT:   cond: <null>
!CHECK-NEXT:   op0 {
!CHECK-NEXT:     what: Write
!CHECK-NEXT:     assign: x=v
!CHECK-NEXT:   }
!CHECK-NEXT:   op1 {
!CHECK-NEXT:     what: None
!CHECK-NEXT:     assign: <null>
!CHECK-NEXT:   }
!CHECK-NEXT: }


subroutine f02(x, v)
  integer :: x, v
  !$omp atomic update
    x = x + v
end

!CHECK: Analysis {
!CHECK-NEXT:   atom: x
!CHECK-NEXT:   cond: <null>
!CHECK-NEXT:   op0 {
!CHECK-NEXT:     what: Update
!CHECK-NEXT:     assign: x=x+v
!CHECK-NEXT:   }
!CHECK-NEXT:   op1 {
!CHECK-NEXT:     what: None
!CHECK-NEXT:     assign: <null>
!CHECK-NEXT:   }
!CHECK-NEXT: }


subroutine f03(x, v)
  integer :: x, v, t
  !$omp atomic update capture
    t = x
    x = x + v
  !$omp end atomic
end

!CHECK: Analysis {
!CHECK-NEXT:   atom: x
!CHECK-NEXT:   cond: <null>
!CHECK-NEXT:   op0 {
!CHECK-NEXT:     what: Read
!CHECK-NEXT:     assign: t=x
!CHECK-NEXT:   }
!CHECK-NEXT:   op1 {
!CHECK-NEXT:     what: Update
!CHECK-NEXT:     assign: x=x+v
!CHECK-NEXT:   }
!CHECK-NEXT: }
