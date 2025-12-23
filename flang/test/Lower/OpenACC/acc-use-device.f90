! This test checks whether the OpenACC use_device clause is applied on both results of hlfir.declare.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Test for automatic variable appearing in use_device clause.
subroutine test()
  integer :: N = 100
  real*8 :: b(-1:N)
! CHECK: %[[A0:.*]] = fir.alloca !fir.array<?xf64>, %{{.*}} {bindc_name = "b", uniq_name = "_QFtestEb"}
! CHECK: %[[A1:.*]] = fir.shape_shift {{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A0]](%[[A1]]) {uniq_name = "_QFtestEb"} : (!fir.ref<!fir.array<?xf64>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)

  !$acc data copy(b)
! CHECK: %[[B:.*]] = acc.copyin var(%[[A]]#0 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK: acc.data dataOperands(%[[B]] : !fir.box<!fir.array<?xf64>>) {

  !$acc host_data use_device(b)
  call vadd(b)
  !$acc end host_data
! CHECK: %[[C:.*]] = acc.use_device var(%[[A]]#0 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {name = "b"}
! CHECK: %[[D:.*]] = acc.use_device varPtr(%[[A]]#1 : !fir.ref<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>> {name = "b"}
! CHECK: acc.host_data dataOperands(%[[C]], %[[D]] : !fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>) {
! CHECK: fir.call @_QPvadd(%[[A]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf64>>) -> ()
  !$acc end data
! CHECK: acc.copyout accVar(%[[B]] : !fir.box<!fir.array<?xf64>>) to var(%[[A]]#0 : !fir.box<!fir.array<?xf64>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
end

! Test for allocatable, pointer and assumed-shape variables appearing in use_device clause.
subroutine test2(a, b, c)
  integer :: N = 100
  real*8, allocatable :: a(:)
  real*8, target, allocatable :: d(:)
  real*8 :: b(:)
  real*8, pointer :: c(:)
  call allocate(a(N))
  call allocate(d(N))
  c => d
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[E:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[DS]] {{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest2Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
! CHECK: %[[F:.*]]:2 = hlfir.declare %arg1 dummy_scope %[[DS]] {{.*}} {uniq_name = "_QFtest2Eb"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
! CHECK: %[[G:.*]]:2 = hlfir.declare %arg2 dummy_scope %[[DS]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest2Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)

  !$acc data copy(a,b,c,d)
  !$acc host_data use_device(a,b,c)
  call vadd2(a,b,c)
  !$acc end host_data

! CHECK: %[[H:.*]] = acc.use_device varPtr(%[[E]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "a"}
! CHECK: %[[I:.*]] = acc.use_device varPtr(%[[E]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "a"}
! CHECK: %[[J:.*]] = acc.use_device var(%[[F]]#0 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {name = "b"}
! CHECK: %[[K:.*]] = acc.use_device var(%[[F]]#1 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {name = "b"}
! CHECK: %[[L:.*]] = acc.use_device varPtr(%[[G]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> {name = "c"}
! CHECK: %[[M:.*]] = acc.use_device varPtr(%[[G]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> {name = "c"}
! CHECK: acc.host_data dataOperands(%[[H]], %[[I]], %[[J]], %[[K]], %[[L]], %[[M]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) {




  !$acc end data

end
