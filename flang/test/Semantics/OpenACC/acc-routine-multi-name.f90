! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc

! Check that !$acc routine(name1, name2) accepts multiple names with a
! warning (extension enabled by default), and that a BIND clause with
! multiple names is an error.

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
  ! Two names: one warning per directive, both symbols get routine info.
  !WARNING: OpenACC ROUTINE directive permits only a single name; multiple names accepted as an extension [-Wopenacc-multiple-names-in-routine]
  !$acc routine(sub1, sub2) seq

  ! Three names: still one warning per directive, not per name.
  !WARNING: OpenACC ROUTINE directive permits only a single name; multiple names accepted as an extension [-Wopenacc-multiple-names-in-routine]
  !$acc routine(sub3, sub4, sub5) gang

  ! Single name: no warning.
  !$acc routine(sub6) seq

  ! BIND clause with multiple names: error.
  !WARNING: OpenACC ROUTINE directive permits only a single name; multiple names accepted as an extension [-Wopenacc-multiple-names-in-routine]
  !ERROR: A BIND clause may only be specified when the ROUTINE directive refers to a single subroutine
  !$acc routine(sub1, sub2) seq bind(sub3)

contains
  subroutine inner()
    ! Unnamed form must be inside a subroutine — no warning.
    !$acc routine seq
  end subroutine
end module
