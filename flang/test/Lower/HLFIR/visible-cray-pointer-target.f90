! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

subroutine visible_association()
  real :: pointee, associated, unrelated
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated)
  pointee = 1.0
  print *, associated, unrelated
end

! CHECK-LABEL: func.func @_QPvisible_association()
! CHECK-DAG: %[[ASSOCIATED_ALLOC:.*]] = fir.alloca f32 {bindc_name = "associated", fir.target, uniq_name = "_QFvisible_associationEassociated"}
! CHECK-DAG: %[[ASSOCIATED:.*]]:2 = hlfir.declare %[[ASSOCIATED_ALLOC]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFvisible_associationEassociated"}
! CHECK-DAG: fir.alloca f32 {bindc_name = "unrelated", uniq_name = "_QFvisible_associationEunrelated"}

subroutine nested_association(flag)
  logical :: flag
  real :: pointee, associated
  integer(8) :: ptr
  pointer(ptr, pointee)
  if (flag) then
    ptr = loc(associated)
  end if
end

! CHECK-LABEL: func.func @_QPnested_association(
! CHECK-DAG: %[[NESTED_ALLOC:.*]] = fir.alloca f32 {bindc_name = "associated", fir.target, uniq_name = "_QFnested_associationEassociated"}
! CHECK-DAG: hlfir.declare %[[NESTED_ALLOC]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFnested_associationEassociated"}

subroutine ordinary_loc()
  real :: object
  integer(8) :: address
  address = loc(object)
end

! CHECK-LABEL: func.func @_QPordinary_loc()
! CHECK: fir.alloca f32 {bindc_name = "object", uniq_name = "_QFordinary_locEobject"}
