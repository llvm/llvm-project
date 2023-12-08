! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-fir -use-alloc-runtime %s -o - | FileCheck %s

! Test global allocatable definition lowering

! CHECK-LABEL: fir.global @_QMmod_allocatablesEc : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {
  ! CHECK-DAG: %[[modcNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK-DAG: %[[modcShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[modcInitBox:.*]] = fir.embox %[[modcNullAddr]](%[[modcShape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.has_value %[[modcInitBox]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>

module mod_allocatables
  character(10), allocatable :: c(:)
end module
  
! CHECK-LABEL: func @_QPtest_mod_allocatables()
subroutine test_mod_allocatables()
  use mod_allocatables, only: c
  ! CHECK: fir.address_of(@_QMmod_allocatablesEc) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  call bar(c(1))
end subroutine


! CHECK-LABEL: func @_QPtest_globals()
subroutine test_globals()
  integer, allocatable :: gx, gy(:, :)
  save :: gx, gy
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgx) : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgy) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  character(:), allocatable :: gc1, gc2(:, :)
  character(10), allocatable :: gc3, gc4(:, :)
  save :: gc1, gc2, gc3, gc4
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgc1) : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgc2) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgc3) : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgc4) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>>
  allocate(gx, gy(20, 30), gc3, gc4(40, 50))
  allocate(character(15):: gc1, gc2(60, 70))
end subroutine

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc1 : !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[gc1NullAddr:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
  ! CHECK: %[[gc1InitBox:.*]] = fir.embox %[[gc1NullAddr]] typeparams %c0{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK: fir.has_value %[[gc1InitBox]] : !fir.box<!fir.heap<!fir.char<1,?>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc2 : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK-DAG: %[[gc2NullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[gc2NullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gc2InitBox:.*]] = fir.embox %[[gc2NullAddr]](%[[gc2NullShape]]) typeparams %c0{{.*}} : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.has_value %[[gc2InitBox]] : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc3 : !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK-DAG: %[[gc3NullAddr:.*]] = fir.zero_bits !fir.heap<!fir.char<1,10>>
  ! CHECK: %[[gc3InitBox:.*]] = fir.embox %[[gc3NullAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[gc3InitBox]] : !fir.box<!fir.heap<!fir.char<1,10>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc4 : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>
  ! CHECK-DAG: %[[gc4NullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
  ! CHECK-DAG: %[[gc4NullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gc4InitBox:.*]] = fir.embox %[[gc4NullAddr]](%[[gc4NullShape]]) : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>
  ! CHECK: fir.has_value %[[gc4InitBox]] : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgx : !fir.box<!fir.heap<i32>>
  ! CHECK: %[[gxNullAddr:.*]] = fir.zero_bits !fir.heap<i32>
  ! CHECK: %[[gxInitBox:.*]] = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: fir.has_value %[[gxInitBox]] : !fir.box<!fir.heap<i32>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgy : !fir.box<!fir.heap<!fir.array<?x?xi32>>> {
  ! CHECK-DAG: %[[gyNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[gyShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gyInitBox:.*]] = fir.embox %[[gyNullAddr]](%[[gyShape]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK: fir.has_value %[[gyInitBox]] : !fir.box<!fir.heap<!fir.array<?x?xi32>>>
