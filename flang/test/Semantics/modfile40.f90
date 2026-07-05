! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that intrinsics in module files retain their 'private' attribute,
! if they are private.

module m1
  intrinsic :: selected_real_kind
  public :: selected_real_kind
end module
!Expect: m1.mod
!module m1
!intrinsic::selected_real_kind
!end

module m2
  use m1, only: foo => selected_real_kind
  real(foo(5,10)) :: x
end module
!Expect: m2.mod
!module m2
!use m1,only:foo=>selected_real_kind
!real(4)::x
!end

module m3
  intrinsic :: selected_real_kind
  private :: selected_real_kind
end module
!Expect: m3.mod
!module m3
!intrinsic::selected_real_kind
!private::selected_real_kind
!end

module m4
  use m3
  external :: selected_real_kind
end module
!Expect: m4.mod
!module m4
!procedure()::selected_real_kind
!end

module m5
  private
  intrinsic :: selected_real_kind
end module
!Expect: m5.mod
!module m5
!intrinsic::selected_real_kind
!private::selected_real_kind
!end

use m2
use m4
use m5
print *, kind(x)
end

