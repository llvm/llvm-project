! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test that a module re-exporting a DEC STRUCTURE writes a parseable module
! file (use m1 rather than use m1,only:/s/).
! https://github.com/llvm/llvm-project/issues/229768

module m1
  structure /s/
    integer :: i
  end structure
  structure /t/
    real :: r
  end structure
end module

module m2
  use m1
end module

!Expect: m1.mod
!module m1
!structure /s/
!integer(4)::i
!end structure
!structure /t/
!real(4)::r
!end structure
!end

!Expect: m2.mod
!module m2
!use m1
!end
