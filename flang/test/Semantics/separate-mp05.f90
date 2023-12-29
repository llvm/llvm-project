! RUN: %python %S/test_symbols.py %s %flang_fc1
! Ensure that SMPs work with dummy procedures declared as interfaces
!DEF: /m Module
module m
 implicit none
 interface
  !DEF: /m/smp MODULE, PUBLIC, PURE (Function) Subprogram REAL(4)
  !DEF: /m/smp/f EXTERNAL, PURE (Function) Subprogram REAL(4)
  !DEF: /m/smp/x INTENT(IN) ObjectEntity REAL(4)
  !DEF: /m/smp/res (Implicit) ObjectEntity REAL(4)
  pure module function smp(f, x) result(res)
   interface
    !REF: /m/smp/f
    !DEF: /m/smp/f/x INTENT(IN) ObjectEntity REAL(4)
    !DEF: /m/smp/f/r ObjectEntity REAL(4)
    pure function f(x) result(r)
     !REF: /m/smp/f/x
     real, intent(in) :: x
     !REF: /m/smp/f/r
     real r
    end function
   end interface
   !REF: /m/smp/x
   real, intent(in) :: x
  end function
 end interface
end module
!REF: /m
!DEF: /m/sm Module
submodule (m)sm
 implicit none
contains
 !DEF: /m/sm/smp MODULE, PUBLIC, PURE (Function) Subprogram REAL(4)
 module procedure smp
  !DEF: /m/sm/smp/res (Implicit) ObjectEntity REAL(4)
  !DEF: /m/sm/smp/f EXTERNAL, PURE (Function) Subprogram REAL(4)
  !DEF: /m/sm/smp/x INTENT(IN) ObjectEntity REAL(4)
  res = f(x)
 end procedure
end submodule
