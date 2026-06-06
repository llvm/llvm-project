!RUN: %flang_fc1 -fdebug-unparse  %s  2>&1 | FileCheck %s
!Test inheritance of implicit rules in submodules and separate module
!procedures.

module m
  implicit integer(1)(a-z)
  interface
    module subroutine mp(da) ! integer(2)
      implicit integer(2)(a-z)
    end
  end interface
  save :: mv ! integer(1)
end

submodule(m) sm1
  implicit integer(8)(a-z)
  save :: sm1v ! integer(8)
  interface
    module subroutine sm1p(da) ! default real
    end
  end interface
end

submodule(m:sm1) sm2
  implicit integer(2)(a-c,e-z)
  save :: sm2v ! integer(2)
 contains
  module subroutine sm1p(da) ! default real
    save :: sm1pv ! inherited integer(2)
    !CHECK: PRINT *, 1_4, 8_4, 2_4, 4_4, 2_4
    print *, kind(mv), kind(sm1v), kind(sm2v), kind(da), kind(sm1pv)
  end
end

submodule(m:sm2) sm3
  implicit integer(8)(a-z)
  save :: sm3v ! integer(8)
 contains
  module procedure mp
    save :: mpv ! inherited integer(8)
    call sm1p(1.)
    !CHECK: PRINT *, 1_4, 8_4, 2_4, 8_4, 2_4, 8_4
    print *, kind(mv), kind(sm1v), kind(sm2v), kind(sm3v), kind(da), kind(mpv)
  end
end

program main
  use m
  call mp(1_2)
end
