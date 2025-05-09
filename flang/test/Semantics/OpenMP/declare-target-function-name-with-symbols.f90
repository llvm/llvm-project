!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s 2>&1 | FileCheck %s

! This used to crash.

module test
  contains
  function ex(a, b, c)
    !$omp declare target(ex)
    integer :: a, b, c
    ex = a + b + c
  end function ex
end module test

!CHECK: !DEF: /test Module
!CHECK: module test
!CHECK: contains
!CHECK:  !DEF: /test/ex PUBLIC (Function, OmpDeclareTarget) Subprogram REAL(4)
!CHECK:  !DEF: /test/ex/a ObjectEntity INTEGER(4)
!CHECK:  !DEF: /test/ex/b ObjectEntity INTEGER(4)
!CHECK:  !DEF: /test/ex/c ObjectEntity INTEGER(4)
!CHECK:  function ex(a, b, c)
!CHECK: !$omp declare target (ex)
!CHECK:   !REF: /test/ex/a
!CHECK:   !REF: /test/ex/b
!CHECK:   !REF: /test/ex/c
!CHECK:   integer a, b, c
!CHECK:   !DEF: /test/ex/ex (Implicit, OmpDeclareTarget) ObjectEntity REAL(4)
!CHECK:   !REF: /test/ex/a
!CHECK:   !REF: /test/ex/b
!CHECK:   !REF: /test/ex/c
!CHECK:   ex = a+b+c
!CHECK:  end function ex
!CHECK: end module test

