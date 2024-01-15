! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module library
contains
  subroutine lib_sub(buf)
!dir$ ignore_tkr(r) buf
    real :: buf(1:*)
  end subroutine
end module

module user
  use library
contains
  subroutine sub(var)
    real :: var(:,:,:)
! CHECK: CALL lib_sub
    call lib_sub(var(1, 2, 3))
  end subroutine
end module
