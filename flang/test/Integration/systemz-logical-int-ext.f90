! Check SystemZ extension attributes on LOGICAL arguments.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-DAG: declare void @cl1(i8 zeroext)
! CHECK-DAG: declare void @cl4(i32 signext)
subroutine s(l1, l4)
  use iso_c_binding
  logical(c_bool) :: l1
  logical(4) :: l4
  interface
    subroutine cl1(a) bind(c, name="cl1")
      import c_bool
      logical(c_bool), value :: a
    end subroutine
    subroutine cl4(a) bind(c, name="cl4")
      logical(4), value :: a
    end subroutine
  end interface
  call cl1(l1)
  call cl4(l4)
end subroutine
