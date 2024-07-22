! Tests delayed privatization for `targets ... private(..)` for allocatables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine target_allocatable(lb, ub, l)
  implicit none
  integer mapped_var
  integer, allocatable :: alloc_var
  real :: real_var

  integer(8) :: lb, ub
  real, dimension(lb:ub) :: real_arr

  complex :: comp_var

  integer(8):: l
  character(len = l)  :: char_var

  !$omp target private(alloc_var, real_var) private(lb, real_arr) &
  !$omp&  private(comp_var) private(char_var)
    mapped_var = 5

    alloc_var = 10
    real_var = 3.14

    real_arr(lb + 1) = 6.28

    comp_var = comp_var * comp_var

    char_var = "hello"
  !$omp end target
end subroutine target_allocatable

! Test the privatizer for `character`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[CHAR_PRIVATIZER_SYM:[^[:space:]]+char_var[^[:space:]]+]]
! CHECK-SAME:   : [[CHAR_TYPE:!fir.boxchar<1>]] alloc {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[CHAR_TYPE]]):
! CHECK-NEXT:   %[[UNBOX:.*]]:2 = fir.unboxchar %[[PRIV_ARG]]
! CHECK:        %[[PRIV_ALLOC:.*]] = fir.alloca !fir.char<1,?>(%[[UNBOX]]#1 : index)
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] typeparams %[[UNBOX]]#1
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : [[CHAR_TYPE]])
! CHECK-NEXT: }

! Test the privatizer for `complex`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[COMP_PRIVATIZER_SYM:[^[:space:]]+comp_var[^[:space:]]+]]
! CHECK-SAME:   : [[COMP_TYPE:!fir.ref<!fir.complex<4>>]] alloc {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[COMP_TYPE]]):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.complex<4>
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : [[COMP_TYPE]])
! CHECK-NEXT: }

! Test the privatizer for `real(:)`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[ARR_PRIVATIZER_SYM:[^[:space:]]+real_arr[^[:space:]]+]]
! CHECK-SAME:   : [[ARR_TYPE:!fir.box<!fir.array<\?xf32>>]] alloc {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[ARR_TYPE]]):
! CHECK:        %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:   %[[DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG]], %[[C0]] : ([[ARR_TYPE]], index)
! CHECK:        %[[PRIV_ALLOCA:.*]] = fir.alloca !fir.array<{{\?}}xf32>
! CHECK-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOCA]](%[[SHAPE_SHIFT]])
! CHECK-NEXT:  omp.yield(%[[PRIV_DECL]]#0 : [[ARR_TYPE]])
! CHECK-NEXT: }

! Test the privatizer for `real(:)`'s lower bound
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[LB_PRIVATIZER_SYM:[^[:space:]]+lb[^[:space:]]+]]
! CHECK-SAME:   : [[LB_TYPE:!fir.ref<i64>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[LB_TYPE]]):
! CHECK-NEXT:   %[[PRIV_ALLOCA:.*]] = fir.alloca i64
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOCA]]
! CHECK-NEXT:  omp.yield(%[[PRIV_DECL]]#0 : [[LB_TYPE]])
! CHECK-NEXT: }

! Test the privatizer for `real`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[REAL_PRIVATIZER_SYM:[^[:space:]]+real_var[^[:space:]]+]]
! CHECK-SAME:   : [[REAL_TYPE:!fir.ref<f32>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[REAL_TYPE]]):
! CHECK-NEXT:   %[[PRIV_ALLOCA:.*]] = fir.alloca f32
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOCA]]
! CHECK-NEXT:  omp.yield(%[[PRIV_DECL]]#0 : [[REAL_TYPE]])
! CHECK-NEXT: }

! Test the privatizer for `allocatable`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[ALLOC_PRIVATIZER_SYM:[^[:space:]]+alloc_var[^[:space:]]+]]
! CHECK-SAME:   : [[ALLOC_TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]] alloc {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[ALLOC_TYPE]]):
! CHECK:        %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi ne, %[[PRIV_ARG_ADDR]], %[[C0]] : i64
!
! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK:          %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32 {fir.must_be_heap = true, {{.*}}}
! CHECK-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[ZERO_BITS]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   }
!
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : [[ALLOC_TYPE]])
!
! CHECK-NEXT: } dealloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[ALLOC_TYPE]]):
!
! CHECK-NEXT:   %[[PRIV_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ADDR:.*]] = fir.box_addr %[[PRIV_VAL]]
! CHECK-NEXT:   %[[PRIV_ADDR_I64:.*]] = fir.convert %[[PRIV_ADDR]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[PRIV_NULL_COND:.*]] = arith.cmpi ne, %[[PRIV_ADDR_I64]], %[[C0]] : i64
!
! CHECK-NEXT:   fir.if %[[PRIV_NULL_COND]] {
! CHECK:          %[[PRIV_VAL_2:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:     %[[PRIV_ADDR_2:.*]] = fir.box_addr %[[PRIV_VAL_2]]
! CHECK-NEXT:     fir.freemem %[[PRIV_ADDR_2]]
! CHECK-NEXT:     %[[ZEROS:.*]] = fir.zero_bits
! CHECK-NEXT:     %[[ZEROS_BOX:.*]]  = fir.embox %[[ZEROS]]
! CHECK-NEXT:     fir.store %[[ZEROS_BOX]] to %[[PRIV_ARG]]
! CHECK-NEXT:   }
!
! CHECK-NEXT:   omp.yield
! CHECK-NEXT: }

! CHECK:      func.func @_QPtarget_allocatable
! CHECK:        %[[MAPPED_ALLOC:.*]] = fir.alloca i32 {bindc_name = "mapped_var", {{.*}}}
! CHECK-NEXT:   %[[MAPPED_DECL:.*]]:2 = hlfir.declare %[[MAPPED_ALLOC]]
! CHECK:        %[[MAPPED_MI:.*]] = omp.map.info var_ptr(%[[MAPPED_DECL]]#1 : !fir.ref<i32>, i32)

! CHECK:        omp.target
! CHECK-SAME:     map_entries(%[[MAPPED_MI]] -> %[[MAPPED_ARG:.*]] : !fir.ref<i32>)
! CHECK-SAME:     private(
! CHECK-SAME:       @[[ALLOC_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[ALLOC_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>,
! CHECK-SAME:       @[[REAL_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[REAL_ARG:.*]] : !fir.ref<f32>,
! CHECK-SAME:       @[[LB_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[LB_ARG:.*]] : !fir.ref<i64>,
! CHECK-SAME:       @[[ARR_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[ARR_ARG:.*]] : !fir.box<!fir.array<?xf32>>,
! CHECK-SAME:       @[[COMP_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[COMP_ARG:.*]] : !fir.ref<!fir.complex<4>>,
! CHECK-SAME:       @[[CHAR_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[CHAR_ARG:.*]] : !fir.boxchar<1>) {
! CHECK-NOT:      fir.alloca
! CHECK:          hlfir.declare %[[MAPPED_ARG]]
! CHECK:          hlfir.declare %[[ALLOC_ARG]]
! CHECK:          hlfir.declare %[[REAL_ARG]]
! CHECK:          hlfir.declare %[[LB_ARG]]
! CHECK:          %[[ARR_ARG_ADDR:.*]] = fir.box_addr %[[ARR_ARG]]
! CHECK:          hlfir.declare %[[ARR_ARG_ADDR]]
! CHECK:          hlfir.declare %[[COMP_ARG]]
! CHECK:          %[[CHAR_ARG_UNBOX:.*]]:2 = fir.unboxchar %[[CHAR_ARG]]
! CHECK:          hlfir.declare %[[CHAR_ARG_UNBOX]]
! CHECK:          omp.terminator
! CHECK-NEXT:   }

