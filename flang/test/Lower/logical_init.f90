! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck --check-prefix=CHECK-UNINT %s
! RUN: %flang_fc1 -emit-fir -finit-logical=true -o - %s | FileCheck --check-prefix=CHECK-TRUE %s
! RUN: %flang_fc1 -emit-fir -finit-logical=false -o - %s | FileCheck --check-prefix=CHECK-FALSE %s

subroutine logical_scalar
!CHECK-UNINT-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-TRUE: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical :: x
end subroutine


subroutine logical_allocatable
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-UINIT-NOT: {{.*}} = fir.convert %true : (i1) -> !fir.logical<4>

!CHECK-TRUE: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical, allocatable :: x
end subroutine


subroutine logical_array
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-UINIT-NOT: {{.*}} = fir.convert %true : (i1) -> !fir.logical<4>

!CHECK-TRUE: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical :: x(5)
end subroutine


subroutine logical_pointer
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-UINIT-NOT: {{.*}} = fir.convert %true : (i1) -> !fir.logical<4>

!CHECK-TRUE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical, pointer :: x
end subroutine


subroutine logical_allocatable_array
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-UINIT-NOT: {{.*}} = fir.convert %true : (i1) -> !fir.logical<4>

!CHECK-TRUE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical, allocatable :: x(:)
end subroutine


subroutine logical_in_equivalence
!CHECK-UNINT-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-UINIT-NOT: {{.*}} = fir.convert %true : (i1) -> !fir.logical<4>

!CHECK-TRUE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK-TRUE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>

!CHECK-FALSE-NOT: {{.*}} = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK-FALSE-NOT: {{.}} = fir.convert %true : (i1) -> !fir.logical<4>
  logical :: x
  real :: y
  equivalence(x,y)
end subroutine
