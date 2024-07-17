! Test lowering of MINLOC intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! simple 1 argument MINLOC
subroutine minloc1(a, s)
  integer :: a(:), s(:)
  s = MINLOC(a)
end subroutine
! CHECK-LABEL: func.func @_QPminloc1(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc with by-ref DIM argument
subroutine minloc2(a, s, d)
  integer :: a(:,:), s(:), d
  s = MINLOC(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPminloc2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 dim %[[DIM]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc with scalar mask argument
subroutine minloc3(a, s, m)
  integer :: a(:), s(:)
  logical :: m
  s = MINLOC(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPminloc3(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.logical<4>>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc with array mask argument
subroutine minloc4(a, s, m)
  integer :: a(:), s(:)
  logical :: m(:)
  s = MINLOC(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPminloc4(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc with all 3 arguments, dim is by-val, array isn't boxed
subroutine minloc5(s)
  integer :: s(2)
  integer :: a(2,2) = reshape((/1, 2, 3, 4/), [2,2])
  s = minloc(a, 1, .true.)
end subroutine
! CHECK-LABEL: func.func @_QPminloc5
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
! CHECK-DAG:     %[[ADDR:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.array<2x2xi32>>
! CHECK-DAG:     %[[ARRAY_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<2>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ADDR]](%[[ARRAY_SHAPE]])
! CHECK-DAG:     %[[OUT_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<1>
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG0]](%[[OUT_SHAPE]])
! CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 dim %[[C1]] mask %[[TRUE]] {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2xi32>>, i32, i1) -> !hlfir.expr<2xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<2xi32>
! CHECK-NEXT:    return
! CHECK-nEXT:  }

! back argument as .true.
subroutine minloc_back(a, s)
  integer :: a(:), s(:)
  s = MINLOC(a, BACK=.TRUE.)
end subroutine
! CHECK-LABEL: func.func @_QPminloc_back(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[C1:.*]] = arith.constant true
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 back %[[C1]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! back argument as logical
subroutine minloc_back2(a, s, b)
  integer :: a(:), s(:)
  logical :: b
  s = MINLOC(a, BACK=b)
end subroutine
! CHECK-LABEL: func.func @_QPminloc_back2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[BACKD:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[BACK:.*]] = fir.load %[[BACKD]]#0 : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 back %[[BACK]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! back argument as optional logical
subroutine minloc_back3(a, s, b)
  integer :: a(:), s(:)
  logical, optional :: b
  s = MINLOC(a, BACK=b)
end subroutine
! CHECK-LABEL: func.func @_QPminloc_back3(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b", fir.optional}) {
! CHECK:        %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-NEXT:   %[[BACKD:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:   %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:   %[[IFP:.*]] = fir.is_present %[[BACKD]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK-NEXT:   %[[BACK:.*]] = fir.if %[[IFP]] -> (!fir.logical<4>) {
! CHECK-NEXT:     %[[IFT:.*]] = fir.load %[[BACKD]]#0 : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:     fir.result %[[IFT]] : !fir.logical<4>
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %false = arith.constant false
! CHECK-NEXT:     %[[IFE:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
! CHECK-NEXT:     fir.result %[[IFE]] : !fir.logical<4>
! CHECK-NEXT:   }
! CHECK-NEXT:   %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 back %[[BACK]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:   hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:   hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi32>
! CHECK-NEXT:   return
! CHECK-NEXT: }


! kind = 2
subroutine minloc_kind(a, s)
  integer :: a(:), s(:)
  s = MINLOC(a, KIND=2)
end subroutine
! CHECK-LABEL: func.func @_QPminloc_kind(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi16>
! CHECK:         %[[ELM:.*]] = hlfir.elemental
! CHECK:         hlfir.assign %[[ELM]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[ELM]] : !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<1xi16>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine minloc6(a, s, d)
  integer, pointer :: d
  integer s(:)
  real :: a(:,:)
  s = minloc(a, (d))
end subroutine
! CHECK-LABEL: func.func @_QPminloc6(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:     %[[DIM_BOX:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:    %[[DIM_ADDR:.*]] = fir.box_addr %[[DIM_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM0:.*]] = fir.load %[[DIM_ADDR]] : !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM1:.*]] = hlfir.no_reassoc %[[DIM0]] : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 dim %[[DIM1]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf32>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! simple 1 argument MINLOC for character
subroutine minloc7(a, s)
  character(*) :: a(:)
  integer :: s(:)
  s = MINLOC(a)
end subroutine
! CHECK-LABEL: func.func @_QPminloc7(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc for character with by-ref DIM argument
subroutine minloc8(a, s, d)
  character(*) :: a(:,:)
  integer :: d, s(:)
  s = MINLOC(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPminloc8(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 dim %[[DIM]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! minloc for character with scalar mask argument
subroutine minloc9(a, s, m)
  character(*) :: a(:)
  integer :: s(:)
  logical :: m
  s = MINLOC(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPminloc9(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.minloc %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.logical<4>>) -> !hlfir.expr<1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine testDynamicallyOptionalMask(array, mask, res)
  integer :: array(:), res(:)
  logical, allocatable :: mask(:)
  res = MINLOC(array, mask=mask)
end subroutine
! CHECK-LABEL: func.func @_QPtestdynamicallyoptionalmask(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[MASK_LOAD:.*]] = fir.load %[[MASK]]#1
! CHECK-NEXT:    %[[MASK_ADDR:.*]] = fir.box_addr %[[MASK_LOAD]]
! CHECK-NEXT:    %[[MASK_ADDR_INT:.*]] = fir.convert %[[MASK_ADDR]]
! CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:    %[[CMP:.*]] = arith.cmpi ne, %[[MASK_ADDR_INT]], %[[C0]] : i64
! it is a shame there is a second load here. The first is generated for
! PreparedActualArgument::isPresent, the second is for optional handling
! CHECK-NEXT:    %[[MASK_LOAD2:.*]] = fir.load %[[MASK]]#1
! CHECK-NEXT:    %[[ABSENT:.*]] = fir.absent !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-NEXT:    %[[SELECT:.*]] = arith.select %[[CMP]], %[[MASK_LOAD2]], %[[ABSENT]]
! CHECK-NEXT:    %[[MINLOC:.*]] = hlfir.minloc %[[ARRAY]]#0 mask %[[SELECT]]
! CHECK-NEXT:    hlfir.assign %[[MINLOC]] to %[[RES]]#0
! CHECK-NEXT:    hlfir.destroy %[[MINLOC]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine testAllocatableArray(array, mask, res)
  integer, allocatable :: array(:)
  integer :: res(:)
  logical :: mask(:)
  res = MINLOC(array, mask=mask)
end subroutine
! CHECK-LABEL: func.func @_QPtestallocatablearray(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[LOADED_ARRAY:.*]] = fir.load %[[ARRAY]]#0
! CHECK-NEXT:    %[[MINLOC:.*]] = hlfir.minloc %[[LOADED_ARRAY]] mask %[[MASK]]#0
! CHECK-NEXT:    hlfir.assign %[[MINLOC]] to %[[RES]]#0
! CHECK-NEXT:    hlfir.destroy %[[MINLOC]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

function testOptionalScalar(array, mask)
  integer :: array(:)
  logical, optional :: mask
  integer :: testOptionalScalar(1)
  testOptionalScalar = minloc(array, mask)
end function
! CHECK-LABEL:   func.func @_QPtestoptionalscalar(
! CHECK-SAME:                                     %[[ARRAY_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"},
! CHECK-SAME:                                     %[[MASK_ARG:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask", fir.optional}) -> !fir.array<1xi32>
! CHECK:           %[[ARRAY_VAR:.*]]:2 = hlfir.declare %[[ARRAY_ARG]]
! CHECK:           %[[MASK_VAR:.*]]:2 = hlfir.declare %[[MASK_ARG]]
! CHECK:           %[[RET_ALLOC:.*]] = fir.alloca !fir.array<1xi32> {bindc_name = "testoptionalscalar", uniq_name = "_QFtestoptionalscalarEtestoptionalscalar"}
! CHECK:           %[[RET_VAR:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK:           %[[MASK_IS_PRESENT:.*]] = fir.is_present %[[MASK_VAR]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:           %[[MASK_BOX:.*]] = fir.embox %[[MASK_VAR]]#1
! CHECK:           %[[ABSENT:.*]] = fir.absent !fir.box<!fir.logical<4>>
! CHECK:           %[[MASK_SELECT:.*]] = arith.select %[[MASK_IS_PRESENT]], %[[MASK_BOX]], %[[ABSENT]]
! CHECK:           %[[RES:.*]] = hlfir.minloc %[[ARRAY_VAR]]#0 mask %[[MASK_SELECT]] {{.*}}: (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.logical<4>>) -> !hlfir.expr<1xi32>
! CHECK:           hlfir.assign %[[RES]] to %[[RET_VAR]]#0
! CHECK:           hlfir.destroy %[[RES]]
! CHECK:           %[[RET:.*]] = fir.load %[[RET_VAR]]#1 : !fir.ref<!fir.array<1xi32>>
! CHECK:           return %[[RET]] : !fir.array<1xi32>
! CHECK:         }

! Test that hlfir.minloc lowering inherits constant
! character length from the argument, when the length
! is unknown from the Fortran::evaluate expression type.
subroutine test_unknown_char_len_result
  character(len=3) :: array(3,3)
  integer :: res(2)
  res = minloc(array(:,:)(:))
end subroutine test_unknown_char_len_result
! CHECK-LABEL:   func.func @_QPtest_unknown_char_len_result() {
! CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[C3_0:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[C3_1:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[ARRAY_ALLOC:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,3>>
! CHECK-DAG:       %[[ARRAY_SHAPE:.*]] = fir.shape %[[C3_0]], %[[C3_1]] : (index, index) -> !fir.shape<2>
! CHECK-DAG:       %[[ARRAY:.*]]:2 = hlfir.declare %[[ARRAY_ALLOC]](%[[ARRAY_SHAPE]]) typeparams %[[C3]]
! CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
! CHECK-DAG:       %[[RES_ALLOC:.*]] = fir.alloca !fir.array<2xi32>
! CHECK-DAG:       %[[RES_SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK-DAG:       %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOC]](%[[RES_SHAPE]])
! CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[C1_3:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[C3_4:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[C1_5:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[C3_6:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[SHAPE:.*]] = fir.shape %[[C3_4]], %[[C3_6]] : (index, index) -> !fir.shape<2>
! CHECK-DAG:       %[[C1_7:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[C3_8:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[C3_9:.*]] = arith.constant 3 : index
! CHECK-DAG:       %[[ARRAY_BOX:.*]] = hlfir.designate %[[ARRAY]]#0 (%[[C1]]:%[[C3_0]]:%[[C1_3]], %[[C1]]:%[[C3_1]]:%[[C1_5]]) substr %[[C1_7]], %[[C3_8]]  shape %[[SHAPE]] typeparams %[[C3_9]]
! CHECK:           %[[EXPR:.*]] = hlfir.minloc %[[ARRAY_BOX]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<3x3x!fir.char<1,3>>>) -> !hlfir.expr<2xi32>
! CHECK-NEXT:      hlfir.assign %[[EXPR]] to %[[RES]]#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
! CHECK-NEXT:      hlfir.destroy %[[EXPR]]
! CHECK-NEXT:      return
! CHECK-NEXT:    }


subroutine scalar_dim1(a, d, m, b, s)
  integer :: a(:), d
  integer :: s(:)
  logical :: m(:), b
  s = MINLOC(a, dim=d, mask=m, kind=2, back=b)
end subroutine
! CHECK-LABEL:  func.func @_QPscalar_dim1(
! CHECK:            %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}, %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}, %[[ARG3:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}, %[[ARG4:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
! CHECK-NEXT:    %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-NEXT:    %[[V0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DSCOPE]]
! CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[DSCOPE]]
! CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[DSCOPE]]
! CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[DSCOPE]]
! CHECK-NEXT:    %[[V4:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %[[DSCOPE]]
! CHECK-NEXT:    %[[V5:.*]] = fir.load %[[V1]]#0 : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    %[[V6:.*]] = fir.load %[[V2]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[V7:.*]] = hlfir.minloc %[[V0]]#0 dim %[[V6]] mask %[[V3]]#0 back %[[V5]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, i32, !fir.box<!fir.array<?x!fir.logical<4>>>, !fir.logical<4>) -> i16
! CHECK-NEXT:    %[[V8:.*]] = fir.convert %[[V7]] : (i16) -> i32
! CHECK-NEXT:    hlfir.assign %[[V8]] to %[[V4]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    return
