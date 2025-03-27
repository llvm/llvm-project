! Test lowering of allocate, deallocate and pointer assignment statements to
! HLFIR.
! RUN: bbc -emit-hlfir -o - %s -I nw | FileCheck %s

subroutine allocation(x)
  character(*), allocatable :: x(:)
  deallocate(x)
  allocate(x(100))
end subroutine
! CHECK-LABEL:   func.func @_QPallocation(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_3]] dummy_scope %[[VAL_1]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFallocationEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclXca783e65b88d3c02cf95fcee70c426bc) : !fir.ref<!fir.char<1,96>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 7 : i32
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_9]], %[[VAL_5]], %[[VAL_6]], %[[VAL_10]], %[[VAL_8]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_12:.*]] = arith.constant false
! CHECK:           %[[VAL_13:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQclXca783e65b88d3c02cf95fcee70c426bc) : !fir.ref<!fir.char<1,96>>
! CHECK:           %[[VAL_15:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_19]], %[[VAL_18]], %[[VAL_20]], %[[VAL_21]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_24:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_22]], %[[VAL_12]], %[[VAL_13]], %[[VAL_23]], %[[VAL_15]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           return
! CHECK:         }

subroutine pointer_assignment(p, ziel)
  real, pointer :: p(:)
  real, target :: ziel(42:)
  p => ziel
  p => ziel(42:77:3)
end subroutine
! CHECK-LABEL:   func.func @_QPpointer_assignment(
! CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
! CHECK-SAME:                                     %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "ziel", fir.target}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFpointer_assignmentEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = fir.shift %[[VAL_5]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) dummy_scope %[[VAL_2]] {fortran_attrs = {{.*}}<target>, uniq_name = "_QFpointer_assignmentEziel"} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_8:.*]] = fir.shift %[[VAL_5]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_9:.*]] = fir.rebox %[[VAL_7]]#1(%[[VAL_8]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_11:.*]] = arith.constant 77 : index
! CHECK:           %[[VAL_12:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_10]]:%[[VAL_11]]:%[[VAL_12]])  shape %[[VAL_14]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<12xf32>>
! CHECK:           %[[VAL_16:.*]] = fir.rebox %[[VAL_15]] : (!fir.box<!fir.array<12xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           return
! CHECK:         }

subroutine pointer_remapping(p, ziel)
  real, pointer :: p(:, :)
  real, target :: ziel(10, 20, 30)
  p(2:7, 3:102) => ziel
end subroutine
! CHECK-LABEL:   func.func @_QPpointer_remapping(
! CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "p"},
! CHECK-SAME:                                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10x20x30xf32>> {fir.bindc_name = "ziel", fir.target}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFpointer_remappingEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 30 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) dummy_scope %[[VAL_2]] {fortran_attrs = {{.*}}<target>, uniq_name = "_QFpointer_remappingEziel"} : (!fir.ref<!fir.array<10x20x30xf32>>, !fir.shape<3>, !fir.dscope) -> (!fir.ref<!fir.array<10x20x30xf32>>, !fir.ref<!fir.array<10x20x30xf32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 7 : i64
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : i64
! CHECK:           %[[VAL_12:.*]] = arith.constant 102 : i64
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_14]] : index
! CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_13]] : index
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_13]] : index
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.array<10x20x30xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_23:.*]] = fir.shape_shift %[[VAL_9]], %[[VAL_17]], %[[VAL_11]], %[[VAL_21]] : (i64, index, i64, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_22]](%[[VAL_23]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_24]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           return
! CHECK:         }

subroutine alloc_comp(x)
  type t
     real, allocatable :: a(:)
  end type
  type(t) :: x(10)
  allocate(x(10_8)%a(100_8))
end subroutine
! CHECK-LABEL:   func.func @_QPalloc_comp(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) dummy_scope %[[VAL_1]] {uniq_name = "_QFalloc_compEx"} : (!fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclXca783e65b88d3c02cf95fcee70c426bc) : !fir.ref<!fir.char<1,96>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 109 : i32
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<10x!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"a"}   {fortran_attrs = {{.*}}<allocatable>} : (!fir.ref<!fir.type<_QFalloc_compTt{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 100 : i64
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_15]], %[[VAL_14]], %[[VAL_16]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_17]], %[[VAL_5]], %[[VAL_6]], %[[VAL_18]], %[[VAL_8]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           return
! CHECK:         }

subroutine ptr_comp_assign(x, ziel)
  type t
     real, pointer :: p(:)
  end type
  type(t) :: x(10)
  real, target :: ziel(100)
  x(9_8)%p => ziel
end subroutine
! CHECK-LABEL:   func.func @_QPptr_comp_assign(
! CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "ziel", fir.target}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFptr_comp_assignEx"} : (!fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) dummy_scope %[[VAL_2]] {fortran_attrs = {{.*}}<target>, uniq_name = "_QFptr_comp_assignEziel"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 9 : index
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<10x!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"p"}   {fortran_attrs = {{.*}}<pointer>} : (!fir.ref<!fir.type<_QFptr_comp_assignTt{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_8]]#1(%[[VAL_12]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           return
! CHECK:         }
