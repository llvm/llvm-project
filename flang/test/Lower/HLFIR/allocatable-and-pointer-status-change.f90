! Test lowering of allocate, deallocate and pointer assignment statements to
! HLFIR.
! RUN: bbc -emit-hlfir -o - %s -I nw | FileCheck %s

subroutine allocation(x)
  character(*), allocatable :: x(:)
! CHECK-LABEL: func.func @_QPallocation(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] typeparams %[[VAL_2:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<allocatable>,  {{.*}}Ex
  deallocate(x)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  fir.freemem %[[VAL_5]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]] = fir.embox %[[VAL_6]](%[[VAL_8]]) typeparams %[[VAL_2]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  allocate(x(100))
! CHECK:  %[[VAL_10:.*]] = arith.constant 100 : i32
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:  %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:  %[[VAL_15:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_2]] : index), %[[VAL_14]] {fir.must_be_heap = true, uniq_name = "_QFallocationEx.alloc"}
! CHECK:  %[[VAL_16:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) typeparams %[[VAL_2]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
! CHECK:  fir.store %[[VAL_17]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
end subroutine

subroutine pointer_assignment(p, ziel)
  real, pointer :: p(:)
  real, target :: ziel(42:)
! CHECK-LABEL: func.func @_QPpointer_assignment(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>,  {{.*}}Ep
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_5:[a-z0-9]*]]) {fortran_attrs = #fir.var_attrs<target>,  {{.*}}Eziel
  p => ziel
! CHECK:  %[[VAL_7:.*]] = fir.shift %[[VAL_4:.*]] : (index) -> !fir.shift<1>
! CHECK:  %[[VAL_8:.*]] = fir.rebox %[[VAL_6]]#1(%[[VAL_7]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_8]] to %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => ziel(42:77:3)
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %{{.*}}#0 (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<12xf32>>
! CHECK:  %[[VAL_15:.*]] = fir.rebox %[[VAL_14]] : (!fir.box<!fir.array<12xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_15]] to %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
end subroutine

subroutine pointer_remapping(p, ziel)
  real, pointer :: p(:, :)
  real, target :: ziel(10, 20, 30)
! CHECK-LABEL: func.func @_QPpointer_remapping(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>,  {{.*}}Ep
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]]) {fortran_attrs = #fir.var_attrs<target>,  {{.*}}Eziel
  p(2:7, 3:102) => ziel
! CHECK:  %[[VAL_8:.*]] = arith.constant 2 : i64
! CHECK:  %[[VAL_9:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_10:.*]] = arith.constant 3 : i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 102 : i64
! CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:  %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_13]] : index
! CHECK:  %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_12]] : index
! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:  %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_17]] : index
! CHECK:  %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_12]] : index
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.array<10x20x30xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_22:.*]] = fir.shape_shift %[[VAL_8]], %[[VAL_16]], %[[VAL_10]], %[[VAL_20]] : (i64, index, i64, index) -> !fir.shapeshift<2>
! CHECK:  %[[VAL_23:.*]] = fir.embox %[[VAL_21]](%[[VAL_22]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:  fir.store %[[VAL_23]] to %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
end subroutine

subroutine alloc_comp(x)
  type t
     real, allocatable :: a(:)
  end type
  type(t) :: x(10)
! CHECK-LABEL: func.func @_QPalloc_comp(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]]) {{.*}}Ex
  allocate(x(10_8)%a(100_8))
! CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_4]])  : (!fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:  %[[VAL_6:.*]] = hlfir.designate %[[VAL_5]]{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_7:.*]] = arith.constant 100 : i64
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : index
! CHECK:  %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : index
! CHECK:  %[[VAL_12:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_11]] {fir.must_be_heap = true, uniq_name = "_QFalloc_compEa.alloc"}
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_14]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
end subroutine

subroutine ptr_comp_assign(x, ziel)
  type t
     real, pointer :: p(:)
  end type
  type(t) :: x(10)
! CHECK-LABEL: func.func @_QPptr_comp_assign(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_3:[a-z0-9]*]]) {{.*}}Ex
  real, target :: ziel(100)
  x(9_8)%p => ziel
! CHECK:  %[[VAL_5:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]]) {fortran_attrs = #fir.var_attrs<target>,  {{.*}}Eziel
! CHECK:  %[[VAL_8:.*]] = arith.constant 9 : index
! CHECK:  %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]])  : (!fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_9]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_12:.*]] = fir.embox %[[VAL_7]]#1(%[[VAL_11]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_12]] to %[[VAL_10]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
end subroutine
