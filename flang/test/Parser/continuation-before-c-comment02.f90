! Continuation before C style comment.
! RUN: rm -rf %t && split-file %s %t
! RUN: %python %S/../Semantics/test_errors.py %t/err01.f90 %flang_fc1 -pedantic -Werror
! RUN: %python %S/../Semantics/test_errors.py %t/err02.f90 %flang_fc1 -fopenmp -pedantic

!--- err01.f90
i&
! ERROR: nonstandard usage: C-style comment [-Wclassic-c-comments]
/* c */ = &
! ERROR: nonstandard usage: C-style comment [-Wclassic-c-comments]
/* d */ 1
end

!--- err02.f90
i&
! ERROR: nonstandard usage: C-style comment [-Wclassic-c-comments]
! ERROR: expected '('
/* c */ &
! ERROR: expected declaration construct
= 2

i&
! ERROR: expected '('
! ERROR: nonstandard usage: C-style comment [-Wclassic-c-comments]
/* c */ & ! d
! ERROR: expected declaration construct
= 3

!$omp parallel do &
! ERROR: nonstandard usage: C-style comment [-Wclassic-c-comments]
/* c */ !$omp private(i)
  do i = 1, 10
  end do
!$omp end parallel do
end
