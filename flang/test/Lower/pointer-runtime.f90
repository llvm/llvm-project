! RUN: bbc -emit-fir -hlfir=false -use-alloc-runtime %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallocate statements.
! CHECK-LABEL: _QPpointer_runtime(
subroutine pointer_runtime(n)
  integer :: n
  character(:), pointer :: scalar, array(:)
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFpointer_runtimeEscalar"}
  ! CHECK-DAG: %[[sNullAddr:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
  ! CHECK-DAG: %[[sInitBox:.*]] = fir.embox %[[sNullAddr]] typeparams %c0{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK-DAG: fir.store %[[sInitBox]] to %[[sBoxAddr]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>

  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFpointer_runtimeEarray"}
  ! CHECK-DAG: %[[aNullAddr:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[aNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[aInitBox:.*]] = fir.embox %[[aNullAddr]](%[[aNullShape]]) typeparams %c0{{.*}} : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK-DAG: fir.store %[[aInitBox]] to %[[aBoxAddr]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>

  allocate(character(10):: scalar, array(30))
  ! CHECK-DAG: %[[sBoxCast1:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten1:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}PointerNullifyCharacter(%[[sBoxCast1]], %[[ten1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-NOT: PointerSetBounds
  ! CHECK: %[[sBoxCast2:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerAllocate(%[[sBoxCast2]]

  ! CHECK-DAG: %[[aBoxCast1:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten2:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}PointerNullifyCharacter(%[[aBoxCast1]], %[[ten2]], %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
  ! CHECK: %[[aBoxCast2:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerSetBounds(%[[aBoxCast2]]
  ! CHECK: %[[aBoxCast3:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerAllocate(%[[aBoxCast3]]

  deallocate(scalar, array)
  ! CHECK: %[[sBoxCast3:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerDeallocate(%[[sBoxCast3]]
  ! CHECK: %[[aBoxCast4:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerDeallocate(%[[aBoxCast4]]

  ! only testing that the correct length is set in the descriptor.
  allocate(character(n):: scalar, array(40))
  ! CHECK: %[[n:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast1:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK-DAG: %[[sBoxCast4:.*]] = fir.convert %[[sBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerNullifyCharacter(%[[sBoxCast4]], %[[ncast1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-DAG: %[[ncast2:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK-DAG: %[[aBoxCast5:.*]] = fir.convert %[[aBoxAddr]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}PointerNullifyCharacter(%[[aBoxCast5]], %[[ncast2]], %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
end subroutine
