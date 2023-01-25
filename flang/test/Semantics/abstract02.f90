! RUN: %python %S/test_errors.py %s %flang_fc1
! Test misuse of abstract interfaces
program test
  abstract interface
    subroutine abstract
    end subroutine
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
