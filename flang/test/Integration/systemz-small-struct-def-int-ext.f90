! Check that a definition gets `noext` on a small BIND(C) struct argument.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-LABEL: define void @cdef(
! CHECK-SAME: i32 noext %{{[^,)]*}})
subroutine cdef(a) bind(c, name="cdef")
  use iso_c_binding
  type, bind(c) :: t32
    integer(c_int32_t) :: i
  end type
  type(t32), value :: a
end subroutine
