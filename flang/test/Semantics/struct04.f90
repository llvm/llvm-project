! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that RECORD /name/ resolves a STRUCTURE brought into scope via USE.
! https://github.com/llvm/llvm-project/issues/228150

module mod_with_structure
  structure /rs/
    integer a /88/
  end structure
  record /rs/ mod_var
end module

subroutine test_use_associated_structure
  use mod_with_structure
  ! No error: /rs/ is USE-associated and must be usable in RECORD.
  record /rs/ local
  local%a = 42
end subroutine

subroutine test_recursive_structure_rejected
  structure /t/
    !ERROR: /t/ is not a known STRUCTURE
    record /t/ nested
    integer n
  end structure
end subroutine
