!RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /m Module
module m
 !DEF: /m/k PUBLIC ObjectEntity INTEGER(4)
 integer k
contains
 !DEF: /m/f PUBLIC, RECURSIVE (Function) Subprogram REAL(4)
 !DEF: /m/f/r ObjectEntity REAL(4)
 recursive function f() result(r)
  !REF: /m/f/r
  real r
  !DEF: /m/e PUBLIC (Function) Subprogram REAL(4)
  !REF: /m/f/r
  entry e() result(r)
  !DEF: /m/f/e (Function) HostAssoc REAL(4)
  !DEF: /m/f/ptr EXTERNAL, POINTER (Function) ProcEntity REAL(4)
  procedure(e), pointer :: ptr => e
  !REF: /m/k
  k = k+1
  !REF: /m/f/r
  r = 20.0
  !REF: /m/k
  if (k==1) then
   !REF: /m/f/ptr
   if (ptr()/=20) error stop
  end if
 end function f
end module
