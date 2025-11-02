! Tests delayed privatization for `targets ... private(..)` for allocatables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization-staging -o - %s 2>&1 \
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
! CHECK: omp.private {type = firstprivate}
! CHECK-SAME: @[[FIRSTPRIVATE_SCALAR_SYM:[^[:space:]]+mapped_var[^[:space:]]+]]
! CHECK-SAME:   : [[FIRSTPRIVATE_TYPE:i32]] copy {

! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[CHAR_PRIVATIZER_SYM:[^[:space:]]+char_var[^[:space:]]+]]
! CHECK-SAME:   : [[CHAR_TYPE:!fir.boxchar<1>]] init {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[CHAR_TYPE]], %[[UNUSED:.*]]: [[CHAR_TYPE]]):
! CHECK-NEXT:   %[[UNBOX:.*]]:2 = fir.unboxchar %[[PRIV_ARG]]
! CHECK:        %[[PRIV_ALLOC:.*]] = fir.allocmem !fir.char<1,?>(%[[UNBOX]]#1 : index)
! CHECK:        %[[BOXCHAR:.*]] = fir.emboxchar %[[PRIV_ALLOC]], %[[UNBOX]]#1
! CHECK-NEXT:   omp.yield(%[[BOXCHAR]] : [[CHAR_TYPE]])
! CHECK-NEXT: } dealloc {

! Test the privatizer for `complex`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[COMP_PRIVATIZER_SYM:[^[:space:]]+comp_var[^[:space:]]+]]
! CHECK-SAME:   : [[COMP_TYPE:complex<f32>]]{{$}}

! Test the privatizer for `real(:)`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[ARR_PRIVATIZER_SYM:[^[:space:]]+real_arr[^[:space:]]+]]
! CHECK-SAME:   : [[ARR_TYPE:!fir.box<!fir.array<\?xf32>>]] init {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<[[ARR_TYPE]]>, %[[PRIV_ALLOC:.*]]: !fir.ref<[[ARR_TYPE]]>):
! CHECK-NEXT:   %[[MOLD:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[MOLD]], %[[C0]]
! CHECK-NEXT:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1
! CHECK-NEXT:   %[[DATA_ALLOC:.*]] = fir.allocmem !fir.array<?xf32>, %[[BOX_DIMS]]#1
! CHECK-NEXT:   %[[DECL:.*]]:2 = hlfir.declare %[[DATA_ALLOC:.*]](%[[SHAPE]])
! CHECK-NEXT:   %[[C0_2:.*]] = arith.constant 0 : index
! CHECK-NEXT:   %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[MOLD]], %[[C0_2]]
! CHECK-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS_2]]#0, %[[BOX_DIMS_2]]#1
! CHECK-NEXT:   %[[BOX:.*]] = fir.rebox %[[DECL]]#0(%[[SHAPE_SHIFT]])
! CHECK-NEXT:   fir.store %[[BOX]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<[[ARR_TYPE]]>)
! CHECK-NEXT: }

! Test the privatizer for `real(:)`'s lower bound
!

! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[LB_PRIVATIZER_SYM:[^[:space:]]+lb[^[:space:]]+]]
! CHECK-SAME:   : [[LB_TYPE:i64]]{{$}}

! Test the privatizer for `real`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[REAL_PRIVATIZER_SYM:[^[:space:]]+real_var[^[:space:]]+]]
! CHECK-SAME:   : [[REAL_TYPE:f32]]{{$}}

! Test the privatizer for `allocatable`
!
! CHECK:      omp.private {type = private}
! CHECK-SAME:   @[[ALLOC_PRIVATIZER_SYM:[^[:space:]]+alloc_var[^[:space:]]+]]
! CHECK-SAME:   : [[ALLOC_TYPE:!fir.box<!fir.heap<i32>>]] init {
!
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<[[ALLOC_TYPE]]>, %[[PRIV_ALLOC:.*]]: !fir.ref<[[ALLOC_TYPE]]>):
! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi eq, %[[PRIV_ARG_ADDR]], %[[C0]] : i64
!
! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   } else {
! CHECK:          %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32
! CHECK-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   }
!
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<[[ALLOC_TYPE]]>)
!
! CHECK-NEXT: } dealloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<[[ALLOC_TYPE]]>):
!
! CHECK-NEXT:   %[[PRIV_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ADDR:.*]] = fir.box_addr %[[PRIV_VAL]]
! CHECK-NEXT:   %[[PRIV_ADDR_I64:.*]] = fir.convert %[[PRIV_ADDR]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[PRIV_NULL_COND:.*]] = arith.cmpi ne, %[[PRIV_ADDR_I64]], %[[C0]] : i64
!
! CHECK-NEXT:   fir.if %[[PRIV_NULL_COND]] {
! CHECK-NEXT:     fir.freemem %[[PRIV_ADDR]]
! CHECK-NEXT:   }
!
! CHECK-NEXT:   omp.yield
! CHECK-NEXT: }

! CHECK:      func.func @_QPtarget_allocatable
! CHECK:        %[[CHAR_VAR_DESC_ALLOCA:.*]] = fir.alloca !fir.boxchar<1>
! CHECK:        %[[REAL_ARR_DESC_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:        %[[ALLOC_VAR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "alloc_var", {{.*}}}
! CHECK:        %[[ALLOC_VAR_DECL:.*]]:2 = hlfir.declare %[[ALLOC_VAR_ALLOCA]]
! CHECK:        %[[MAPPED_ALLOC:.*]] = fir.alloca i32 {bindc_name = "mapped_var", {{.*}}}
! CHECK-NEXT:   %[[MAPPED_DECL:.*]]:2 = hlfir.declare %[[MAPPED_ALLOC]]
! CHECK:        %[[CHAR_VAR_ALLOC:.*]] = fir.alloca !fir.char<1,?>{{.*}} {bindc_name = "char_var", {{.*}}}
! CHECK:        %[[CHAR_VAR_DECL:.*]]:2 = hlfir.declare %[[CHAR_VAR_ALLOC]] typeparams
! CHECK:        %[[REAL_ARR_ALLOC:.*]] = fir.alloca !fir.array<?xf32>, {{.*}} {bindc_name = "real_arr", {{.*}}}
! CHECK:        %[[REAL_ARR_DECL:.*]]:2 = hlfir.declare %[[REAL_ARR_ALLOC]]({{.*}})
! CHECK:        fir.store %[[REAL_ARR_DECL]]#0 to %[[REAL_ARR_DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:        %[[ALLOC_VAR_MEMBER:.*]] = omp.map.info var_ptr(%[[ALLOC_VAR_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>, i32)
! CHECK:        %[[ALLOC_VAR_MAP:.*]] = omp.map.info var_ptr(%[[ALLOC_VAR_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) {{.*}} members(%[[ALLOC_VAR_MEMBER]] :
! CHECK:        %[[REAL_ARR_MEMBER:.*]] = omp.map.info var_ptr(%[[REAL_ARR_DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, f32)
! CHECK:        %[[REAL_ARR_DESC_MAP:.*]] = omp.map.info var_ptr(%[[REAL_ARR_DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.box<!fir.array<?xf32>>) {{.*}} members(%[[REAL_ARR_MEMBER]] :
! CHECK:        fir.store %[[CHAR_VAR_DECL]]#0 to %[[CHAR_VAR_DESC_ALLOCA]] : !fir.ref<!fir.boxchar<1>>
! CHECK:        %[[CHAR_VAR_DESC_MAP:.*]] = omp.map.info var_ptr(%[[CHAR_VAR_DESC_ALLOCA]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>)
! CHECK:        %[[MAPPED_MI0:.*]] = omp.map.info var_ptr(%[[MAPPED_DECL]]#0 : !fir.ref<i32>, i32) {{.*}}
! CHECK:        omp.target
! CHECK-SAME:     map_entries(
! CHECK-SAME:       %[[ALLOC_VAR_MAP]] -> %[[MAPPED_ARG1:[^,]+]]
! CHECK-SAME:       %[[REAL_ARR_DESC_MAP]] -> %[[MAPPED_ARG2:[^,]+]]
! CHECK-SAME:       %[[CHAR_VAR_DESC_MAP]] -> %[[MAPPED_ARG3:.[^,]+]]
! CHECK-SAME:       %[[MAPPED_MI0]] -> %[[MAPPED_ARG0:[^,]+]]
! CHECK-SAME:       !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.ref<!fir.boxchar<1>>, !fir.ref<i32>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.boxchar<1>>
! CHECK-SAME:     private(
! CHECK-SAME:       @[[ALLOC_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[ALLOC_ARG:[^,]+]] [map_idx=0],
! CHECK-SAME:       @[[REAL_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[REAL_ARG:[^,]+]],
! CHECK-SAME:       @[[LB_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[LB_ARG:[^,]+]],
! CHECK-SAME:       @[[ARR_PRIVATIZER_SYM]] %{{[^[:space:]]+}} -> %[[ARR_ARG:[^,]+]] [map_idx=1],
! CHECK-SAME:       @[[COMP_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[COMP_ARG:[^,]+]],
! CHECK-SAME:       @[[CHAR_PRIVATIZER_SYM]] %{{[^[:space:]]+}}#0 -> %[[CHAR_ARG:[^,]+]] [map_idx=2]
! CHECK-SAME:       @[[FIRSTPRIVATE_SCALAR_SYM]] %{{[^[:space:]]+}}#0 -> %[[FP_SCALAR_ARG:[^,]+]] [map_idx=3] :
! CHECK-SAME:       !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<f32>, !fir.ref<i64>, !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.ref<complex<f32>>, !fir.boxchar<1>, !fir.ref<i32>)
! CHECK-NOT:      fir.alloca
! CHECK:          hlfir.declare %[[ALLOC_ARG]]
! CHECK:          hlfir.declare %[[REAL_ARG]]
! CHECK:          hlfir.declare %[[LB_ARG]]
! CHECK:          hlfir.declare %[[ARR_ARG]]
! CHECK:          hlfir.declare %[[COMP_ARG]]
! CHECK:          %[[CHAR_ARG_UNBOX:.*]]:2 = fir.unboxchar %[[CHAR_ARG]]
! CHECK:          hlfir.declare %[[CHAR_ARG_UNBOX]]
! CHECK:          omp.terminator
! CHECK-NEXT:   }

