!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! Test that empty parent types are still set first in the
! runtime info global array describing components.
module empty_parent
 type :: z
 end type

 type, extends(z) :: t
  integer :: a
 end type
end module
! CHECK: .c.t, SAVE{{.*}}.n.z{{.*}}n.a
