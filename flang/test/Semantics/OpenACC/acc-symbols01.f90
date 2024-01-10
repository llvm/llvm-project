! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenacc

!DEF: /mm MainProgram
program mm
  !DEF: /mm/x ObjectEntity REAL(4)
  !DEF: /mm/y ObjectEntity REAL(4)
  real x, y
  !DEF: /mm/a ObjectEntity INTEGER(4)
  !DEF: /mm/b ObjectEntity INTEGER(4)
  !DEF: /mm/c ObjectEntity INTEGER(4)
  !DEF: /mm/i ObjectEntity INTEGER(4)
  integer a(10), b(10), c(10), i
  !REF: /mm/b
  b = 2
 !$acc parallel present(c) firstprivate(b) private(a)
 !$acc loop
  !REF: /mm/i
  do i=1,10
   !REF: /mm/a
   !REF: /mm/i
   !REF: /mm/b
   a(i) = b(i)
  end do
 !$acc end parallel
 end program

