! RUN: bbc -emit-fir -use-alloc-runtime %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
! CHECK-LABEL: _QPfoo
subroutine foo()
    real, allocatable :: x(:), y(:, :), z
    ! CHECK: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}uniq_name = "_QFfooEx"}
    ! CHECK-DAG: %[[xNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    ! CHECK-DAG: %[[xNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
    ! CHECK: %[[xInitEmbox:.*]] = fir.embox %[[xNullAddr]](%[[xNullShape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    ! CHECK: fir.store %[[xInitEmbox]] to %[[xBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  
    ! CHECK: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {{{.*}}uniq_name = "_QFfooEy"}
    ! CHECK-DAG: %[[yNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
    ! CHECK-DAG: %[[yNullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
    ! CHECK: %[[yInitEmbox:.*]] = fir.embox %[[yNullAddr]](%[[yNullShape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    ! CHECK: fir.store %[[yInitEmbox]] to %[[yBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  
    ! CHECK: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {{{.*}}uniq_name = "_QFfooEz"}
    ! CHECK: %[[zNullAddr:.*]] = fir.zero_bits !fir.heap<f32>
    ! CHECK: %[[zInitEmbox:.*]] = fir.embox %[[zNullAddr]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
    ! CHECK: fir.store %[[zInitEmbox]] to %[[zBoxAddr]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  
    allocate(x(42:100), y(43:50, 51), z)
    ! CHECK-DAG: %[[errMsg:.*]] = fir.absent !fir.box<none>
    ! CHECK-DAG: %[[xlb:.*]] = arith.constant 42 : i32
    ! CHECK-DAG: %[[xub:.*]] = arith.constant 100 : i32
    ! CHECK-DAG: %[[xBoxCast2:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    ! CHECK-DAG: %[[xlbCast:.*]] = fir.convert %[[xlb]] : (i32) -> i64
    ! CHECK-DAG: %[[xubCast:.*]] = fir.convert %[[xub]] : (i32) -> i64
    ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[xBoxCast2]], %c0{{.*}}, %[[xlbCast]], %[[xubCast]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
    ! CHECK-DAG: %[[xBoxCast3:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    ! CHECK-DAG: %[[sourceFile:.*]] = fir.convert %{{.*}} -> !fir.ref<i8>
    ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[xBoxCast3]], %false{{.*}}, %[[errMsg]], %[[sourceFile]], %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  
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
    ! CHECK: %[[yBounds1:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
    ! CHECK: %[[yBounds2:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
    ! CHECK: %[[yAddr:.*]] = fir.box_addr %[[yBoxLoad]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
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
    ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_deferredEscalar"}
    ! CHECK-DAG: %[[sNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
    ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] typeparams %c0{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  
    ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_deferredEarray"}
    ! CHECK-DAG: %[[aNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
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
    ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEscalar"}
    ! CHECK-DAG: %[[sNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.char<1,10>>
    ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
    ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  
    ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEarray"}
    ! CHECK-DAG: %[[aNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,10>>>
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
    ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEscalar"}
    ! CHECK-DAG: %[[sNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
    ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] typeparams %[[l1]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  
    character(l2), allocatable :: array(:)
    ! CHECK-DAG: %[[l2:.*]] = fir.load %arg2 : !fir.ref<i32>
    ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEarray"}
    ! CHECK-DAG: %[[aNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
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
