! Test lowering of derived type descriptor for types with
! a default initialized character component.

! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

subroutine test()
  type t
    character(5) :: character_comp = "hello"
  end type
  type(t) :: x
end subroutine
! CHECK-LABEL: fir.global {{.*}} @_QFtestE.c.t constant
! CHECK: fir.address_of(@_QFtestE.di.t.character_comp) : !fir.ref<!fir.char<1,5>>
