! RUN: %python %S/test_modfile.py %s %flang_fc1

! Cray pointer
module m
  integer :: pte
  pointer(ptr,pte)
  integer :: apte
  pointer(aptr,apte(7))
end

!Expect: m.mod
!module m
!integer(4)::pte
!integer(8)::ptr
!pointer(ptr,pte)
!integer(4)::apte(1_8:7_8)
!integer(8)::aptr
!pointer(aptr,apte)
!end
