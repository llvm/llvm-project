! Check SystemZ extension attributes on CHARACTER(1) passed by value.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-DAG: declare zeroext i8 @cc(i8 zeroext)
! CHECK-DAG: define zeroext i8 @f(i8 zeroext %{{.*}})
subroutine s(c)
  use iso_c_binding
  character(c_char) :: c
  interface
    function cc(a) bind(c, name="cc")
      import c_char
      character(c_char) :: cc
      character(c_char), value :: a
    end function
  end interface
  c = cc(c)
end subroutine

function f(a) bind(c, name="f")
  use iso_c_binding
  character(c_char) :: f
  character(c_char), value :: a
  f = a
end function
