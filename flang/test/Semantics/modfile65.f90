! RUN: %python %S/test_modfile.py %s %flang_fc1 -fhermetic-module-files
module m1
  integer, parameter :: n = 123
end

module m2
  use m1
end

module m3
  use m1, m => n
end

module m4
  use m2
  use m3
end

!Expect: m1.mod
!module m1
!integer(4),parameter::n=123_4
!end

!Expect: m2.mod
!module m2
!use m1,only:n
!end
!module m1
!integer(4),parameter::n=123_4
!end

!Expect: m3.mod
!module m3
!use m1,only:m=>n
!end
!module m1
!integer(4),parameter::n=123_4
!end

!Expect: m4.mod
!module m4
!use m2,only:n
!use m3,only:m
!end
!module m2
!use m1,only:n
!end
!module m3
!use m1,only:m=>n
!end
!module m1
!integer(4),parameter::n=123_4
!end
