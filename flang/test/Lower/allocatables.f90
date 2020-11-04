! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of allocatables
! CHECK-LABEL: _QPfoo
subroutine foo()
  real, allocatable :: x(:), y(:, :), z
  ! CHECK: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {name = "_QFfooEx"}
  ! CHECK-DAG: %[[xTypeCat:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[xKind:.*]] = constant 4 : i64
  ! CHECK-DAG: %[[xRank:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[xCorank:.*]] = constant 0 : i32
  ! CHECK-DAG: %[[xBoxCast:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[xKindCast:.*]] = fir.convert %[[xKind]] : (i64) -> i32
  ! CHECK: fir.call @{{.*}}AllocatableInitIntrinsic(%[[xBoxCast]], %[[xTypeCat]], %[[xKindCast]], %[[xRank]], %[[xCorank]]) : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none

  ! CHECK: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {name = "_QFfooEy"}
  ! CHECK-DAG: %[[yTypeCat:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[yKind:.*]] = constant 4 : i64
  ! CHECK-DAG: %[[yRank:.*]] = constant 2 : i32
  ! CHECK-DAG: %[[yCorank:.*]] = constant 0 : i32
  ! CHECK-DAG: %[[yBoxCast:.*]] = fir.convert %[[yBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[yKindCast:.*]] = fir.convert %[[yKind]] : (i64) -> i32
  ! CHECK: fir.call @{{.*}}AllocatableInitIntrinsic(%[[yBoxCast]], %[[yTypeCat]], %[[yKindCast]], %[[yRank]], %[[yCorank]]) : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none

  ! CHECK: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {name = "_QFfooEz"}
  ! CHECK-DAG: %[[zTypeCat:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[zKind:.*]] = constant 4 : i64
  ! CHECK-DAG: %[[zRank:.*]] = constant 0 : i32
  ! CHECK-DAG: %[[zCorank:.*]] = constant 0 : i32
  ! CHECK-DAG: %[[zBoxCast:.*]] = fir.convert %[[zBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[zKindCast:.*]] = fir.convert %[[zKind]] : (i64) -> i32
  ! CHECK: fir.call @{{.*}}AllocatableInitIntrinsic(%[[zBoxCast]], %[[zTypeCat]], %[[zKindCast]], %[[zRank]], %[[zCorank]]) : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none

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
