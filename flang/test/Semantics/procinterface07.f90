! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/sm1.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/sm2.f90
! C1515: an interface name declared by a procedure-declaration-stmt in an
! ancestor module or submodule that was compiled separately is declared
! before any use in a descendant, so these uses must be accepted.

!--- m.f90
module procinterface07m
  abstract interface
    subroutine s_iface()
    end subroutine
  end interface
  procedure(s_iface), pointer :: q
  interface
    module subroutine ms()
    end subroutine
  end interface
end module

!--- sm1.f90
submodule (procinterface07m) procinterface07sm1
  procedure(q), pointer :: r1
  procedure(s_iface), pointer :: q1
end submodule

!--- sm2.f90
submodule (procinterface07m:procinterface07sm1) procinterface07sm2
  procedure(q1), pointer :: r2
contains
  module subroutine ms()
    procedure(q), pointer :: r3
    procedure(q1), pointer :: r4
    r3 => null()
    r4 => null()
  end subroutine
end submodule
