! RUN: %python %S/test_errors.py %s %flang_fc1
! Test misuse of abstract interfaces
program test
  abstract interface
    subroutine abstract
    end subroutine
    !ERROR: An ABSTRACT interface may not have the same name as an intrinsic type
    function integer()
    end
    !ERROR: An ABSTRACT interface may not have the same name as an intrinsic type
    subroutine logical
    end
  end interface
  procedure(abstract), pointer :: p
  !ERROR: Abstract procedure interface 'abstract' may not be referenced
  call abstract
  !ERROR: Abstract procedure interface 'abstract' may not be used as a designator
  p => abstract
  !ERROR: Abstract procedure interface 'abstract' may not be used as a designator
  call foo(abstract)
  !ERROR: Abstract procedure interface 'abstract' may not be used as a designator
  print *, associated(p, abstract)
end
