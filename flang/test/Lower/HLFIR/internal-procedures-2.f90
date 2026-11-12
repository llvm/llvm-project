! Test instantiation of module variables inside an internal subprogram
! where the use statement is inside the host program.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module module_used_by_host
 implicit none
 integer :: indexed_by_var(2)
 integer :: ref_in_implied_do
 integer :: ref_in_forall(2)
end module

subroutine host_procedure
 use module_used_by_host
 implicit none
contains
 subroutine internal_procedure(i, mask)
  integer :: i
  logical :: mask(2)
  indexed_by_var(i) = 0
  print *, (/(ref_in_implied_do, integer::j=1,10)/)
  forall (integer::k = 1:2)
    ref_in_forall(k) = 0
  end forall
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFhost_procedurePinternal_procedure(
! CHECK:    fir.address_of(@_QMmodule_used_by_hostEindexed_by_var) : !fir.ref<!fir.array<2xi32>>
! CHECK:    fir.address_of(@_QMmodule_used_by_hostEref_in_forall) : !fir.ref<!fir.array<2xi32>>
! CHECK:    fir.address_of(@_QMmodule_used_by_hostEref_in_implied_do) : !fir.ref<i32>
