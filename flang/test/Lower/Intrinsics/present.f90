! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: present_test
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?xi32>>
subroutine present_test(a)
  integer, optional :: a(:)

  if (present(a)) print *,a
  ! CHECK: %{{.*}} = fir.is_present %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> i1
end subroutine
