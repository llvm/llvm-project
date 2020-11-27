! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of allocatables
! CHECK-LABEL: _QPfoo
subroutine foo()
  real, allocatable :: x(:), y(:, :), z
  ! CHECK: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {name = "_QFfooEx"}
  ! CHECK-DAG: %[[xNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[xNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[xInitEmbox:.*]] = fir.embox %[[xNullAddr]](%[[xNullShape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[xInitEmbox]] to %[[xBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>

  ! CHECK: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {name = "_QFfooEy"}
  ! CHECK-DAG: %[[yNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG: %[[yNullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[yInitEmbox:.*]] = fir.embox %[[yNullAddr]](%[[yNullShape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[yInitEmbox]] to %[[yBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

  ! CHECK: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {name = "_QFfooEz"}
  ! CHECK: %[[zNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<f32>
  ! CHECK: %[[zInitEmbox:.*]] = fir.embox %[[zNullAddr]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! CHECK: fir.store %[[zInitEmbox]] to %[[zBoxAddr]] : !fir.ref<!fir.box<!fir.heap<f32>>>


  allocate(x(42:100), y(43:50, 51), z)
  ! CHECK-DAG: %[[xlb:.*]] = constant 42 : i32
  ! CHECK-DAG: %[[xub:.*]] = constant 100 : i32
  ! CHECK-DAG: %[[xBoxCast2:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[xlbCast:.*]] = fir.convert %[[xlb]] : (i32) -> i64
  ! CHECK-DAG: %[[xubCast:.*]] = fir.convert %[[xub]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[xBoxCast2]], %c0{{.*}}, %[[xlbCast]], %[[xubCast]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
  ! CHECK-DAG: %[[xBoxCast3:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[errMsg:.*]] = fir.convert %{{.*}} : (!fir.ref<none>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[sourceFile:.*]] = fir.convert %{{.*}} -> !fir.ref<i8>
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[xBoxCast3]], %false{{.*}}, %[[errMsg]], %[[sourceFile]], %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.ref<!fir.box<none>>, !fir.ref<i8>, i32) -> i32

  ! Simply check that we are emitting the right numebr of set bound for y and z. Otherwise, this is just like x.
  ! CHECK: fir.convert %[[yBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate
  ! CHECK: %[[zBoxCast:.*]] = fir.convert %[[zBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-NOT: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate

  ! Check that y descriptor is read when referencing it.
  ! CHECK: %[[yBoxLoad:.*]] = fir.load %[[yBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: %[[yAddr:.*]] = fir.box_addr %[[yBoxLoad]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK: %[[yBounds1:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[yBounds2:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  print *, x, y(45, 46), z

  deallocate(x, y, z)
  ! CHECK: %[[xBoxCast4:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[xBoxCast4]], {{.*}})
  ! CHECK: %[[yBoxCast4:.*]] = fir.convert %[[yBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[yBoxCast4]], {{.*}})
  ! CHECK: %[[zBoxCast4:.*]] = fir.convert %[[zBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[zBoxCast4]], {{.*}})
end subroutine

! test lowering of character allocatables
! CHECK-LABEL: _QPchar_deferred(
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {name = "_QFchar_deferredEscalar"}
  ! CHECK-DAG: %[[sNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.char<1,?>>
  ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] typeparams %c0{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {name = "_QFchar_deferredEarray"}
  ! CHECK-DAG: %[[aNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[aNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[aInitBox:.*]] = fir.embox %[[aNullAddr]](%[[aNullShape]]) typeparams %c0{{.*}} : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK-DAG: fir.store %[[aInitBox]] to %[[aBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>

  allocate(character(10):: scalar, array(30))
  ! CHECK-DAG: %[[sBoxCast1:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten1:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacter(%[[sBoxCast1]], %[[ten1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-NOT: AllocatableSetBounds
  ! CHECK: %[[sBoxCast2:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[sBoxCast2]]

  ! CHECK-DAG: %[[aBoxCast1:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten2:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacter(%[[aBoxCast1]], %[[ten2]], %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
  ! CHECK: %[[aBoxCast2:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[aBoxCast2]]
  ! CHECK: %[[aBoxCast3:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[aBoxCast3]]

  deallocate(scalar, array)
  ! CHECK: %[[sBoxCast3:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[sBoxCast3]]
  ! CHECK: %[[aBoxCast4:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[aBoxCast4]]

  ! only testing that the correct length is set in the descriptor.
  allocate(character(n):: scalar, array(40))
  ! CHECK: %[[n:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast1:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK-DAG: %[[sBoxCast4:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacter(%[[sBoxCast4]], %[[ncast1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-DAG: %[[ncast2:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK-DAG: %[[aBoxCast5:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacter(%[[aBoxCast5]], %[[ncast2]], %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
end subroutine

! CHECK-LABEL: _QPchar_explicit_cst(
subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {name = "_QFchar_explicit_cstEscalar"}
  ! CHECK-DAG: %[[sNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.char<1,10>>
  ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>

  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {name = "_QFchar_explicit_cstEarray"}
  ! CHECK-DAG: %[[aNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK-DAG: %[[aNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[aInitBox:.*]] = fir.embox %[[aNullAddr]](%[[aNullShape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK-DAG: fir.store %[[aInitBox]] to %[[aBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  allocate(scalar, array(20))
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  deallocate(scalar, array)
  ! CHECK: AllocatableDeallocate
  ! CHECK: AllocatableDeallocate
end subroutine

! CHECK-LABEL: _QPchar_explicit_dyn(
subroutine char_explicit_dyn(n, l1, l2)
  integer :: n, l1, l2
  character(l1), allocatable :: scalar
  ! CHECK-DAG: %[[l1:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {name = "_QFchar_explicit_dynEscalar"}
  ! CHECK-DAG: %[[sNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.char<1,?>>
  ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] typeparams %[[l1]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

  character(l2), allocatable :: array(:)
  ! CHECK-DAG: %[[l2:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {name = "_QFchar_explicit_dynEarray"}
  ! CHECK-DAG: %[[aNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[aNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[aInitBox:.*]] = fir.embox %[[aNullAddr]](%[[aNullShape]]) typeparams %[[l2]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK-DAG: fir.store %[[aInitBox]] to %[[aBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  allocate(scalar, array(20))
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  deallocate(scalar, array)
  ! CHECK: AllocatableDeallocate
  ! CHECK: AllocatableDeallocate
end subroutine

!  Test global allocatables lowering 

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
  ! CHECK-DAG: %[[gc1NullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.char<1,?>>
  ! CHECK: %[[gc1InitBox:.*]] = fir.embox %[[gc1NullAddr]] typeparams %c0{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK: fir.has_value %[[gc1InitBox]] : !fir.box<!fir.heap<!fir.char<1,?>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc2 : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK-DAG: %[[gc2NullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[gc2NullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gc2InitBox:.*]] = fir.embox %[[gc2NullAddr]](%[[gc2NullShape]]) typeparams %c0{{.*}} : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.has_value %[[gc2InitBox]] : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc3 : !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK-DAG: %[[gc3NullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.char<1,10>>
  ! CHECK: %[[gc3InitBox:.*]] = fir.embox %[[gc3NullAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[gc3InitBox]] : !fir.box<!fir.heap<!fir.char<1,10>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgc4 : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>
  ! CHECK-DAG: %[[gc4NullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
  ! CHECK-DAG: %[[gc4NullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gc4InitBox:.*]] = fir.embox %[[gc4NullAddr]](%[[gc4NullShape]]) : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>
  ! CHECK: fir.has_value %[[gc4InitBox]] : !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgx : !fir.box<!fir.heap<i32>>
  ! CHECK: %[[gxNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<i32>
  ! CHECK: %[[gxInitBox:.*]] = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: fir.has_value %[[gxInitBox]] : !fir.box<!fir.heap<i32>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgy : !fir.box<!fir.heap<!fir.array<?x?xi32>>> {
  ! CHECK-DAG: %[[gyNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[gyShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gyInitBox:.*]] = fir.embox %[[gyNullAddr]](%[[gyShape]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK: fir.has_value %[[gyInitBox]] : !fir.box<!fir.heap<!fir.array<?x?xi32>>>

