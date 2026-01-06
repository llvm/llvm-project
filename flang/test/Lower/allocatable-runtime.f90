! RUN: %flang_fc1 -emit-fir -O2 -mllvm -use-alloc-runtime %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
! CHECK-LABEL: _QPfoo
subroutine foo()
  real, allocatable :: x(:), y(:, :), z
  ! CHECK-DAG: %[[xlb:.*]] = arith.constant 42 : i32
  ! CHECK-DAG: %[[xub:.*]] = arith.constant 100 : i32
  ! CHECK-DAG: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}uniq_name = "_QFfooEx"}
  ! CHECK-DAG: %[[xNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[xInitEmbox:.*]] = fir.embox %[[xNullAddr]]
  ! CHECK-DAG: fir.store %[[xInitEmbox]] to %[[xBoxAddr]]
  ! CHECK-DAG: %[[xBoxDecl:.*]] = fir.declare %[[xBoxAddr]]{{.*}}

  ! CHECK-DAG: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {{{.*}}uniq_name = "_QFfooEy"}
  ! CHECK-DAG: %[[yNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG: %[[yInitEmbox:.*]] = fir.embox %[[yNullAddr]]
  ! CHECK-DAG: fir.store %[[yInitEmbox]] to %[[yBoxAddr]]
  ! CHECK-DAG: %[[yBoxDecl:.*]] = fir.declare %[[yBoxAddr]]{{.*}}

  ! CHECK-DAG: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {{{.*}}uniq_name = "_QFfooEz"}
  ! CHECK-DAG: %[[zNullAddr:.*]] = fir.zero_bits !fir.heap<f32>
  ! CHECK-DAG: %[[zInitEmbox:.*]] = fir.embox %[[zNullAddr]]
  ! CHECK-DAG: fir.store %[[zInitEmbox]] to %[[zBoxAddr]]
  ! CHECK-DAG: %[[zBoxDecl:.*]] = fir.declare %[[zBoxAddr]]{{.*}}

  allocate(x(42:100), y(43:50, 51), z)
  ! CHECK-DAG: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK-DAG: %[[xBoxCast2:.*]] = fir.convert %[[xBoxDecl]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[xlbCast:.*]] = fir.convert %[[xlb]] : (i32) -> i64
  ! CHECK-DAG: %[[xubCast:.*]] = fir.convert %[[xub]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[xBoxCast2]], %c0{{.*}}, %[[xlbCast]], %[[xubCast]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[xBoxCast2]], %{{.*}}, %false{{.*}}, %[[errMsg]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32

  ! Simply check that we are emitting the right numebr of set bound for y and z. Otherwise, this is just like x.
  ! CHECK: %[[yBoxCast:.*]] = fir.convert %[[yBoxDecl]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate
  ! CHECK: %[[zBoxCast:.*]] = fir.convert %[[zBoxDecl]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-NOT: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate

  ! Check that y descriptor is read when referencing it.
  ! CHECK: %[[yBoxLoad:.*]] = fir.load %[[yBoxDecl]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK-DAG: %[[yAddr:.*]] = fir.box_addr %[[yBoxLoad]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG: %[[yBounds1:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[yBounds2:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  print *, x, y(45, 46), z

  deallocate(x, y, z)
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[xBoxCast2]], {{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[yBoxCast]], {{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[zBoxCast]], {{.*}})
end subroutine

! test lowering of character allocatables
! CHECK-LABEL: _QPchar_deferred(
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[nArgDecl:.*]] = fir.declare %arg0 {{.*}}
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_deferredEscalar"}
  ! CHECK-DAG: %[[sBoxDecl:.*]] = fir.declare %[[sBoxAddr]]{{.*}}
  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_deferredEarray"}
  ! CHECK-DAG: %[[aBoxDecl:.*]] = fir.declare %[[aBoxAddr]]{{.*}}

  allocate(character(10):: scalar, array(30))
  ! CHECK-DAG: %[[sBoxCast1:.*]] = fir.convert %[[sBoxDecl]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten1:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[sBoxCast1]], %[[ten1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-NOT: AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[sBoxCast1]]

  ! CHECK-DAG: %[[aBoxCast1:.*]] = fir.convert %[[aBoxDecl]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[aBoxCast1]], %{{.*}}, %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[aBoxCast1]]
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[aBoxCast1]]

  deallocate(scalar, array)
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[sBoxCast1]], {{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[aBoxCast1]], {{.*}})

  ! only testing that the correct length is set in the descriptor.
  allocate(character(n):: scalar, array(40))
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[nArgDecl]] : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast1:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[sBoxCast1]], %[[ncast1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[aBoxCast1]], %[[ncast1]], %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
end subroutine

! CHECK-LABEL: _QPchar_explicit_cst(
subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEscalar"}
  ! CHECK-DAG: %[[sBoxDecl:.*]] = fir.declare %[[sBoxAddr]]{{.*}}
  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEarray"}
  ! CHECK-DAG: %[[aBoxDecl:.*]] = fir.declare %[[aBoxAddr]]{{.*}}

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
  ! CHECK-DAG:  %[[l1Decl:.*]] = fir.declare %arg1 {{.*}}
  ! CHECK-DAG:  %[[l2Decl:.*]] = fir.declare %arg2 {{.*}}
  ! CHECK-DAG:  %[[c0_i32:.*]] = arith.constant 0 : i32
  ! CHECK-DAG:  %[[raw_l1:.*]] = fir.load %[[l1Decl]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[cmp1:.*]] = arith.cmpi sgt, %[[raw_l1]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[l1:.*]] = arith.select %[[cmp1]], %[[raw_l1]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEscalar"}
  ! CHECK-DAG:  %[[sBoxDecl:.*]] = fir.declare %[[sBoxAddr]]{{.*}}

  character(l2), allocatable :: zarray(:)
  ! CHECK-DAG:  %[[raw_l2:.*]] = fir.load %[[l2Decl]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[cmp2:.*]] = arith.cmpi sgt, %[[raw_l2]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[l2:.*]] = arith.select %[[cmp2]], %[[raw_l2]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEzarray"}
  ! CHECK-DAG:  %[[aBoxDecl:.*]] = fir.declare %[[aBoxAddr]]{{.*}}

  allocate(scalar, zarray(20))
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  deallocate(scalar, zarray)
  ! CHECK: AllocatableDeallocate
  ! CHECK: AllocatableDeallocate
end subroutine

subroutine mold_allocation()
  integer :: m(10)
  integer, allocatable :: a(:)

  allocate(a, mold=m)
end subroutine

! CHECK-LABEL: func.func @_QPmold_allocation() {
! CHECK-DAG: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFmold_allocationEa"}
! CHECK-DAG: %[[M:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "m", uniq_name = "_QFmold_allocationEm"}
! CHECK-DAG: %[[M_DECL:.*]] = fir.declare %[[M]]{{.*}}
! CHECK-DAG: %[[A_DECL:.*]] = fir.declare %[[A]]{{.*}}
! CHECK-DAG: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[EMBOX_M:.*]] = fir.embox %[[M_DECL]](%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[M_BOX_NONE:.*]] = fir.convert %[[EMBOX_M]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %[[M_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[A_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32
