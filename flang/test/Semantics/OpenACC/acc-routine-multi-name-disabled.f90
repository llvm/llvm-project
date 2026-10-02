! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc -fno-openacc-multiple-names-in-routine

! Check that -fno-openacc-multiple-names-in-routine turns the multiple-names
! extension into a hard error. A BIND clause with multiple names does not
! produce an additional diagnostic when the extension is disabled.

subroutine sub1()
end subroutine

subroutine sub2()
end subroutine

subroutine sub3()
end subroutine

subroutine sub4()
end subroutine

subroutine sub5()
end subroutine

subroutine sub6()
end subroutine

module m
  ! Two names with BIND: only the multiple-names error fires when disabled.
  !ERROR: OpenACC ROUTINE directive does not permit multiple names
  !$acc routine(sub1, sub2) seq bind(sub3)

  ! Three names: still one error per directive.
  !ERROR: OpenACC ROUTINE directive does not permit multiple names
  !$acc routine(sub3, sub4, sub5) gang

  ! Single name: no error.
  !$acc routine(sub6) seq

contains
  subroutine inner()
    ! Unnamed form must be inside a subroutine — no error.
    !$acc routine seq
  end subroutine
end module
