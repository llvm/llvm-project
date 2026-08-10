! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module library
contains
  subroutine lib_sub(buf)
!dir$ ignore_tkr(c) buf
    real :: buf(1:*)
  end subroutine
end module

module user
  use library
contains
  subroutine sub(var, ptr)
    real :: var(:,:,:)
    real, pointer :: ptr(:)
! CHECK: CALL lib_sub
    call lib_sub(var(1, 2, 3))
! CHECK: CALL lib_sub
    call lib_sub(ptr(1))
  end subroutine
end module
