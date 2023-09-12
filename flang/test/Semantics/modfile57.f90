! RUN: %python %S/test_modfile.py %s %flang_fc1

! Cray pointer
module m
  integer :: pte
  pointer(ptr,pte)
end

!Expect: m.mod
!module m
!integer(4)::pte
!integer(8)::ptr
!pointer(ptr,pte)
!end
