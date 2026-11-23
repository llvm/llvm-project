! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine close_member_no_parent()
  type t
    integer :: x, y
  end type
  type(t) :: s

! CHECK-LABEL: func.func @_QPclose_member_no_parent()
! CHECK-NOT: map_clauses(close
! CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("s%x") -> !fir.ref<i32>
! CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(storage) {{.*}} members(%[[MEMBER]] : [0] : !fir.ref<i32>) name("s") partial_map(true)
! CHECK-NOT: map_clauses(close
  !$omp target map(close, tofrom: s%x)
  s%x = s%y
  !$omp end target
end

subroutine close_member_close_parent()
  type t
    integer :: x, y
  end type
  type(t) :: s

! CHECK-LABEL: func.func @_QPclose_member_close_parent()
! CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(close, tofrom) {{.*}} name("s%x") -> !fir.ref<i32>
! CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(close, tofrom) {{.*}} members(%[[MEMBER]] : [0] : !fir.ref<i32>) name("s")
  !$omp target map(close, tofrom: s) map(close, tofrom: s%x)
  s%x = s%y
  !$omp end target
end

subroutine close_member_nonclose_parent()
  type t
    integer :: x, y
  end type
  type(t) :: s

! CHECK-LABEL: func.func @_QPclose_member_nonclose_parent()
! CHECK-NOT: map_clauses(close
! CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("s%x") -> !fir.ref<i32>
! CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} members(%[[MEMBER]] : [0] : !fir.ref<i32>) name("s")
! CHECK-NOT: map_clauses(close
  !$omp target map(tofrom: s) map(close, tofrom: s%x)
  s%x = s%y
  !$omp end target
end
