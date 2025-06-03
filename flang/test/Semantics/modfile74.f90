! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
 contains
  function f() result(ptr)
    character :: str
    pointer(ptr, str)
    ptr = 0
  end
end

!Expect: m.mod
!module m
!contains
!function f() result(ptr)
!integer(8)::ptr
!pointer(ptr,str)
!character(1_8,1)::str
!end
!end
