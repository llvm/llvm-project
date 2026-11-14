! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Names in the data clauses of a DECLARE directive that name resolution
! leaves without a symbol must be reported once and must not crash the
! duplicate-occurrence checks.

module acc_declare_unresolved
  implicit none
  real :: a(10)
contains

  subroutine module_name
    !ERROR: No explicit type declared for 'acc_declare_unresolved'
    !$acc declare present(acc_declare_unresolved)
  end subroutine

  subroutine each_clause
    !ERROR: No explicit type declared for 'u1'
    !$acc declare present(u1)
    !ERROR: No explicit type declared for 'u2'
    !$acc declare copy(u2)
    !ERROR: No explicit type declared for 'u3'
    !$acc declare copyin(u3)
    !ERROR: No explicit type declared for 'u4'
    !$acc declare copyout(u4)
    !ERROR: No explicit type declared for 'u5'
    !$acc declare create(u5)
    !ERROR: No explicit type declared for 'u6'
    !$acc declare deviceptr(u6)
    !ERROR: No explicit type declared for 'u7'
    !$acc declare device_resident(u7)
    !ERROR: No explicit type declared for 'u8'
    !$acc declare link(u8)
  end subroutine

  subroutine intrinsic_name
    !ERROR: No explicit type declared for 'sin'
    !$acc declare present(sin)
  end subroutine

  subroutine own_name
    !ERROR: Only variables are allowed in data clauses on the DECLARE directive
    !$acc declare present(own_name)
  end subroutine

  integer function no_result()
    !$acc declare present(no_result)
    no_result = 1
  end function

  integer function with_result() result(res)
    !ERROR: Only variables are allowed in data clauses on the DECLARE directive
    !$acc declare present(with_result)
    res = 1
  end function

  subroutine valid_after_unresolved
    real :: b(10)
    !ERROR: No explicit type declared for 'u9'
    !$acc declare create(u9, b)
    !ERROR: 'b' in the COPYIN clause is already present in another CREATE clause in this module
    !$acc declare copyin(b)
  end subroutine

end module

subroutine external_unit
  implicit none
  !ERROR: No explicit type declared for 'u10'
  !$acc declare present(u10)
end subroutine
