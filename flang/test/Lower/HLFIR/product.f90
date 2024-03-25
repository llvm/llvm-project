! Test lowering of PRODUCT intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! simple 1 argument PRODUCT
subroutine product1(a, s)
  integer :: a(:), s
  s = PRODUCT(a)
end subroutine
! CHECK-LABEL: func.func @_QPproduct1(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! product with by-ref DIM argument
subroutine product2(a, s, d)
  integer :: a(:,:), s(:), d
  s = PRODUCT(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPproduct2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 dim %[[DIM]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! product with scalar mask argument
subroutine product3(a, s, m)
  integer :: a(:), s
  logical :: m
  s = PRODUCT(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPproduct3(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "m"})
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.logical<4>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! product with array mask argument
subroutine product4(a, s, m)
  integer :: a(:), s
  logical :: m(:)
  s = PRODUCT(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPproduct4(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"})
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! product with all 3 arguments, dim is by-val, array isn't boxed
subroutine product5(s)
  integer :: s(2)
  integer :: a(2,2) = reshape((/1, 2, 3, 4/), [2,2])
  s = PRODUCT(a, 1, .true.)
end subroutine
! CHECK-LABEL: func.func @_QPproduct5(
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
! CHECK-DAG:     %[[ADDR:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.array<2x2xi32>>
! CHECK-DAG:     %[[ARRAY_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<2>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ADDR]](%[[ARRAY_SHAPE]])
! CHECK-DAG:     %[[OUT_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<1>
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG0]](%[[OUT_SHAPE]])
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 dim %[[C1]] mask %[[TRUE]]  {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2xi32>>, i32, i1) -> !hlfir.expr<2xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<2xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! product with dimesnsion from pointer
subroutine product6(a, s, d)
  integer, pointer :: d
  real :: a(:,:), s(:)
  s = PRODUCT(a, (d))
end subroutine
! CHECK-LABEL: func.func @_QPproduct6(
! CHECK:           %[[ARG0:.*]]:  !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "d"})
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM_BOX:.*]] = fir.load %[[DIM]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:    %[[DIM_ADDR:.*]] = fir.box_addr %[[DIM_BOX]]  : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM0:.*]] = fir.load %[[DIM_ADDR]] : !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM1:.*]] = hlfir.no_reassoc %[[DIM0]] : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.product %[[ARRAY]]#0 dim %[[DIM1]]  {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf32>>, i32) -> !hlfir.expr<?xf32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine testDynamicallyOptionalMask(array, mask, res)
  integer :: array(:), res
  logical, allocatable :: mask(:)
  res = PRODUCT(array, mask=mask)
end subroutine
! CHECK-LABEL: func.func @_QPtestdynamicallyoptionalmask(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[MASK_LOAD:.*]] = fir.load %[[MASK]]#1
! CHECK-NEXT:    %[[MASK_ADDR:.*]] = fir.box_addr %[[MASK_LOAD]]
! CHECK-NEXT:    %[[MASK_ADDR_INT:.*]] = fir.convert %[[MASK_ADDR]]
! CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:    %[[CMP:.*]] = arith.cmpi ne, %[[MASK_ADDR_INT]], %[[C0]] : i64
! CHECK-NEXT:    %[[MASK_LOAD2:.*]] = fir.load %[[MASK]]#1
! CHECK-NEXT:    %[[ABSENT:.*]] = fir.absent !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-NEXT:    %[[SELECT:.*]] = arith.select %[[CMP]], %[[MASK_LOAD2]], %[[ABSENT]]
! CHECK-NEXT:    %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]]#0 mask %[[SELECT]]
! CHECK-NEXT:    hlfir.assign %[[PRODUCT]] to %[[RES]]#0
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine testAllocatableArray(array, mask, res)
  integer, allocatable :: array(:)
  integer :: res
  logical :: mask(:)
  res = PRODUCT(array, mask=mask)
end subroutine
! CHECK-LABEL: func.func @_QPtestallocatablearray(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[LOADED_ARRAY:.*]] = fir.load %[[ARRAY]]#0
! CHECK-NEXT:    %[[PRODUCT:.*]] = hlfir.product %[[LOADED_ARRAY]] mask %[[MASK]]#0
! CHECK-NEXT:    hlfir.assign %[[PRODUCT]] to %[[RES]]#0
! CHECK-NEXT:    return
! CHECK-NEXT:  }

function testOptionalScalar(array, mask)
  integer :: array(:)
  logical, optional :: mask
  integer :: testOptionalScalar
  testOptionalScalar = product(array, mask)
end function
! CHECK-LABEL:   func.func @_QPtestoptionalscalar(
! CHECK-SAME:                                     %[[ARRAY_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"},
! CHECK-SAME:                                     %[[MASK_ARG:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask", fir.optional}) -> i32
! CHECK:           %[[ARRAY_VAR:.*]]:2 = hlfir.declare %[[ARRAY_ARG]]
! CHECK:           %[[MASK_VAR:.*]]:2 = hlfir.declare %[[MASK_ARG]]
! CHECK:           %[[RET_ALLOC:.*]] = fir.alloca i32 {bindc_name = "testoptionalscalar", uniq_name = "_QFtestoptionalscalarEtestoptionalscalar"}
! CHECK:           %[[RET_VAR:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK:           %[[MASK_IS_PRESENT:.*]] = fir.is_present %[[MASK_VAR]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:           %[[MASK_BOX:.*]] = fir.embox %[[MASK_VAR]]#1
! CHECK:           %[[ABSENT:.*]] = fir.absent !fir.box<!fir.logical<4>>
! CHECK:           %[[MASK_SELECT:.*]] = arith.select %[[MASK_IS_PRESENT]], %[[MASK_BOX]], %[[ABSENT]]
! CHECK:           %[[RES:.*]] = hlfir.product %[[ARRAY_VAR]]#0 mask %[[MASK_SELECT]] {{.*}}: (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.logical<4>>) -> i32
! CHECK:           hlfir.assign %[[RES]] to %[[RET_VAR]]#0
! CHECK:           %[[RET:.*]] = fir.load %[[RET_VAR]]#1 : !fir.ref<i32>
! CHECK:           return %[[RET]] : i32
! CHECK:         }