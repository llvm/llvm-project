! RUN: %python %S/test_errors.py %s %flang_fc1
! Procedure characteristics retain PROTECTED_TARGET on dummies and results.

module characteristics
  integer, target :: storage
contains
  subroutine read(a)
    integer, pointer, protected_target :: a
  end subroutine
  subroutine write(a)
    integer, pointer :: a
  end subroutine
  function read_result() result(p)
    integer, pointer, protected_target :: p
    p => storage
  end function
  function write_result() result(p)
    integer, pointer :: p
    p => storage
  end function

  subroutine test
    procedure(read), pointer :: reader
    procedure(write), pointer :: writer
    procedure(read_result), pointer :: read_function
    procedure(write_result), pointer :: write_function
    reader => read
    writer => write
    read_function => read_result
    write_function => write_result
    !ERROR: Procedure pointer 'reader' associated with incompatible procedure designator 'write': incompatible dummy argument #1: incompatible dummy data object attributes: ProtectedTarget
    reader => write
    !ERROR: Procedure pointer 'writer' associated with incompatible procedure designator 'read': incompatible dummy argument #1: incompatible dummy data object attributes: ProtectedTarget
    writer => read
    !ERROR: Function pointer 'read_function' associated with incompatible function designator 'write_result': function results have incompatible attributes
    read_function => write_result
    !ERROR: Function pointer 'write_function' associated with incompatible function designator 'read_result': function results have incompatible attributes
    write_function => read_result
  end subroutine
end module
