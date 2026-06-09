! RUN: %flang_fc1 -emit-hlfir -O2 -mllvm -use-alloc-runtime %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
! CHECK-LABEL: func.func @_QPfoo
subroutine foo()
  real, allocatable :: x(:), y(:, :), z
  ! CHECK-DAG: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}uniq_name = "_QFfooEx"}
  ! CHECK-DAG: %[[xNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[xInitEmbox:.*]] = fir.embox %[[xNullAddr]]{{.*}}
  ! CHECK-DAG: fir.store %[[xInitEmbox]] to %[[xBoxAddr]]
  ! CHECK-DAG: %[[xBoxDecl:.*]]:2 = hlfir.declare %[[xBoxAddr]]{{.*}}

  ! CHECK-DAG: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {{{.*}}uniq_name = "_QFfooEy"}
  ! CHECK-DAG: %[[yNullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG: %[[yInitEmbox:.*]] = fir.embox %[[yNullAddr]]{{.*}}
  ! CHECK-DAG: fir.store %[[yInitEmbox]] to %[[yBoxAddr]]
  ! CHECK-DAG: %[[yBoxDecl:.*]]:2 = hlfir.declare %[[yBoxAddr]]{{.*}}

  ! CHECK-DAG: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {{{.*}}uniq_name = "_QFfooEz"}
  ! CHECK-DAG: %[[zNullAddr:.*]] = fir.zero_bits !fir.heap<f32>
  ! CHECK-DAG: %[[zInitEmbox:.*]] = fir.embox %[[zNullAddr]]
  ! CHECK-DAG: fir.store %[[zInitEmbox]] to %[[zBoxAddr]]
  ! CHECK-DAG: %[[zBoxDecl:.*]]:2 = hlfir.declare %[[zBoxAddr]]{{.*}}

  allocate(x(42:100), y(43:50, 51), z)
  ! CHECK-DAG: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK-DAG: %[[xlb:.*]] = arith.constant 42 : i32
  ! CHECK-DAG: %[[xub:.*]] = arith.constant 100 : i32
  ! CHECK-DAG: %[[xBoxCast2:.*]] = fir.convert %[[xBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[xlbCast:.*]] = fir.convert %[[xlb]] : (i32) -> i64
  ! CHECK-DAG: %[[xubCast:.*]] = fir.convert %[[xub]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[xBoxCast2]], %c0{{.*}}, %[[xlbCast]], %[[xubCast]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%{{.*}}, %{{.*}}, %false{{.*}}, %[[errMsg]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32

  ! Simply check that we are emitting the right numebr of set bound for y and z. Otherwise, this is just like x.
  ! CHECK: %[[yBoxCast:.*]] = fir.convert %[[yBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate
  ! CHECK: %[[zBoxCast:.*]] = fir.convert %[[zBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-NOT: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate

  ! Check that y descriptor is read when referencing it.
  ! CHECK: %[[yBoxLoad:.*]] = fir.load %[[yBoxDecl]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: hlfir.designate %[[yBoxLoad]] (%{{.*}}, %{{.*}})  : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index, index) -> !fir.ref<f32>
  print *, x, y(45, 46), z

  deallocate(x, y, z)
  ! CHECK: %[[xBoxCastDealloc:.*]] = fir.convert %[[xBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[xBoxCastDealloc]], {{.*}})
  ! CHECK: %[[yBoxCastDealloc:.*]] = fir.convert %[[yBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[yBoxCastDealloc]], {{.*}})
  ! CHECK: %[[zBoxCastDealloc:.*]] = fir.convert %[[zBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[zBoxCastDealloc]], {{.*}})
end subroutine

! test lowering of character allocatables
! CHECK-LABEL: func.func @_QPchar_deferred(
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[nArgDecl:.*]]:2 = hlfir.declare %arg0 {{.*}}
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_deferredEscalar"}
  ! CHECK-DAG: %[[sBoxDecl:.*]]:2 = hlfir.declare %[[sBoxAddr]]{{.*}}
  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_deferredEarray"}
  ! CHECK-DAG: %[[aBoxDecl:.*]]:2 = hlfir.declare %[[aBoxAddr]]{{.*}}

  allocate(character(10):: scalar, array(30))
  ! CHECK-DAG: %[[sBoxCast1:.*]] = fir.convert %[[sBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[ten1:.*]] = fir.convert %c10{{.*}} : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[sBoxCast1]], %[[ten1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK-NOT: AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%{{.*}}

  ! CHECK-DAG: %[[aBoxCast1:.*]] = fir.convert %[[aBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%[[aBoxCast1]], %{{.*}}, %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%{{.*}}
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%{{.*}}

  deallocate(scalar, array)
  ! CHECK: %[[sBoxCastDealloc:.*]] = fir.convert %[[sBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[sBoxCastDealloc]], {{.*}})
  ! CHECK: %[[aBoxCastDealloc:.*]] = fir.convert %[[aBoxDecl]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[aBoxCastDealloc]], {{.*}})

  ! only testing that the correct length is set in the descriptor.
  allocate(character(n):: scalar, array(40))
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[nArgDecl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast1:.*]] = fir.convert %[[n]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%{{.*}}, %[[ncast1]], %c1{{.*}}, %c0{{.*}}, %c0{{.*}})
  ! CHECK: fir.call @{{.*}}AllocatableInitCharacterForAllocate(%{{.*}}, %{{.*}}, %c1{{.*}}, %c1{{.*}}, %c0{{.*}})
end subroutine

! CHECK-LABEL: func.func @_QPchar_explicit_cst(
subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: scalar, array(:)
  ! CHECK-DAG: %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEscalar"}
  ! CHECK-DAG: %[[sBoxDecl:.*]]:2 = hlfir.declare %[[sBoxAddr]]{{.*}}
  ! CHECK-DAG: %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEarray"}
  ! CHECK-DAG: %[[aBoxDecl:.*]]:2 = hlfir.declare %[[aBoxAddr]]{{.*}}

  allocate(scalar, array(20))
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  ! CHECK-NOT: AllocatableInitCharacter
  ! CHECK: AllocatableAllocate
  deallocate(scalar, array)
  ! CHECK: AllocatableDeallocate
  ! CHECK: AllocatableDeallocate
end subroutine

! CHECK-LABEL: func.func @_QPchar_explicit_dyn(
subroutine char_explicit_dyn(n, l1, l2)
  integer :: n, l1, l2
  character(l1), allocatable :: scalar
  ! CHECK-DAG:  %[[l1Decl:.*]]:2 = hlfir.declare %arg1 {{.*}}
  ! CHECK-DAG:  %[[l2Decl:.*]]:2 = hlfir.declare %arg2 {{.*}}
  ! CHECK-DAG:  %[[c0_i32:.*]] = arith.constant 0 : i32
  ! CHECK-DAG:  %[[raw_l1:.*]] = fir.load %[[l1Decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG:  %[[cmp1:.*]] = arith.cmpi sgt, %[[raw_l1]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[l1:.*]] = arith.select %[[cmp1]], %[[raw_l1]], %[[c0_i32]] : i32
  ! CHECK-DAG:  %[[sBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEscalar"}
  ! CHECK-DAG:  %[[sBoxDecl:.*]]:2 = hlfir.declare %[[sBoxAddr]]{{.*}}

  character(l2), allocatable :: zarray(:)
  ! CHECK-DAG:  %[[raw_l2:.*]] = fir.load %[[l2Decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG:  %[[cmp2:.*]] = arith.cmpi sgt, %[[raw_l2]], %c0{{.*}} : i32
  ! CHECK-DAG:  %[[l2:.*]] = arith.select %[[cmp2]], %[[raw_l2]], %c0{{.*}} : i32
  ! CHECK-DAG:  %[[aBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEzarray"}
  ! CHECK-DAG:  %[[aBoxDecl:.*]]:2 = hlfir.declare %[[aBoxAddr]]{{.*}}

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
! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[M]]{{.*}}
! CHECK-DAG: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]{{.*}}
! CHECK: %[[EMBOX_M:.*]] = fir.embox %[[M_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[M_BOX_NONE:.*]] = fir.convert %[[EMBOX_M]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %[[M_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32
