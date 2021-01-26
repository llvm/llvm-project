! RUN: %S/../test_symbols.sh %s %t %f18 -fopenacc

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
  !DEF: /mm/Block1/i (AccPrivate, AccPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
   !DEF: /mm/Block1/a (AccPrivate) HostAssoc INTEGER(4)
   !REF: /mm/Block1/i
   !DEF: /mm/Block1/b (AccFirstPrivate) HostAssoc INTEGER(4)
   a(i) = b(i)
  end do
 !$acc end parallel
 end program

