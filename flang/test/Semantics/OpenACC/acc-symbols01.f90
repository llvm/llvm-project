! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenacc

!DEF: /MM MainProgram
program MM
  !DEF: /MM/x ObjectEntity REAL(4)
  !DEF: /MM/y ObjectEntity REAL(4)
  real x, y
  !DEF: /MM/a ObjectEntity INTEGER(4)
  !DEF: /MM/b ObjectEntity INTEGER(4)
  !DEF: /MM/c ObjectEntity INTEGER(4)
  !DEF: /MM/i ObjectEntity INTEGER(4)
  integer a(10), b(10), c(10), i
  !REF: /MM/b
  b = 2
 !$acc parallel present(c) firstprivate(b) private(a)
 !$acc loop
  !REF: /MM/i
  do i=1,10
   !REF: /MM/a
   !REF: /MM/i
   !REF: /MM/b
   a(i) = b(i)
  end do
 !$acc end parallel
 end program

