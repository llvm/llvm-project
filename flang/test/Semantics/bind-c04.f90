! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1520
! If proc-language-binding-spec (bind(c)) with NAME= is specified, then
! proc-decl-list shall contain exactly one proc-decl, which shall neither have
! the POINTER attribute nor be a dummy procedure.

subroutine sub(x, y)

  interface
    subroutine proc() bind(c)
    end
  end interface

  !Acceptable (as an extension)
  procedure(proc), bind(c, name="aaa") :: pc1, pc2

  !ERROR: BIND(C) procedure with NAME= specified can neither have POINTER attribute nor be a dummy procedure
  procedure(proc), bind(c, name="bbb"), pointer :: pc3

  !ERROR: BIND(C) procedure with NAME= specified can neither have POINTER attribute nor be a dummy procedure
  procedure(proc), bind(c, name="ccc") :: x

  procedure(proc), bind(c) :: pc4, pc5

  !ERROR: BIND(C) procedure with NAME= specified can neither have POINTER attribute nor be a dummy procedure
  procedure(proc), bind(c, name="pc6"), pointer :: pc6

  procedure(proc), bind(c), pointer :: pc7

  procedure(proc), bind(c) :: y

  !WARNING: Attribute 'BIND(C)' cannot be used more than once
  !ERROR: BIND(C) procedure with NAME= specified can neither have POINTER attribute nor be a dummy procedure
  procedure(proc), bind(c, name="pc8"), bind(c), pointer :: pc8

end
