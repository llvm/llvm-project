! This test checks lowering of OpenACC cache directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {

! CHECK-LABEL: func.func @_QPtest_cache_basic()
subroutine test_cache_basic()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b"
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_basicEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_readonly()
subroutine test_cache_readonly()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(readonly: b)
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_cache_readonly>, name = "b"
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_readonlyEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_array_section()
! For b(2:5) with startIdx=1: lowerbound = 2-1 = 1, upperbound = 5-1 = 4, extent = 4
subroutine test_cache_array_section()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(2:5))
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C2:.*]] = arith.constant 2 : index
! CHECK: %[[C5:.*]] = arith.constant 5 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C2]], %[[C1]] : index
! CHECK: %[[TMP1:.*]] = arith.subi %[[C5]], %[[C1]] : index
! CHECK: %[[TMP2:.*]] = arith.subi %[[TMP1]], %[[LB]] : index
! CHECK: %[[EXT:.*]] = arith.addi %[[TMP2]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_array_sectionEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_multiple()
subroutine test_cache_multiple()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b, c)
    a(i) = b(i) + c(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE_B:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b"
! CHECK: hlfir.declare %[[CACHE_B]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEb"}
! CHECK: %[[CACHE_C:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "c"
! CHECK: hlfir.declare %[[CACHE_C]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEc"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_2d_array()
! For b(1:5, 1:5) with startIdx=1: each dimension has lowerbound = 1-1 = 0, upperbound = 5-1 = 4, extent = 5
subroutine test_cache_2d_array()
  integer, parameter :: n = 10
  real, dimension(n, n) :: a, b
  integer :: i, j

  !$acc loop
  do i = 1, n
    !$acc cache(b(1:5, 1:5))
    do j = 1, n
      a(i,j) = b(i,j)
    end do
  end do

! CHECK: acc.loop
! CHECK-DAG: arith.constant 1 : index
! CHECK-DAG: arith.constant 5 : index
! Dimension 1: lowerbound = 1 - startIdx = 0, upperbound = 5 - startIdx = 4, extent = 5
! CHECK: %[[LB1:.*]] = arith.subi %{{.*}}, %{{.*}} : index
! CHECK: arith.subi
! CHECK: arith.subi
! CHECK: arith.addi
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! Dimension 2: lowerbound = 1 - startIdx = 0, upperbound = 5 - startIdx = 4, extent = 5
! CHECK: %[[LB2:.*]] = arith.subi %{{.*}}, %{{.*}} : index
! CHECK: arith.subi
! CHECK: arith.subi
! CHECK: arith.addi
! CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_2d_arrayEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_loop_var()
! Test cache with loop variable dependent bounds: b(i:i+2)
subroutine test_cache_loop_var()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n-2
    !$acc cache(b(i:i+2))
    a(i) = b(i) + b(i+1) + b(i+2)
  end do

! CHECK: acc.loop private({{.*}}) control(%[[IV:.*]] : i32) = ({{.*}}) to ({{.*}})
! The privatized iterator is declared and initialized from the loop control variable
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_loop_varEi"}
! CHECK: fir.store %[[IV]] to %[[I_DECL]]#0 : !fir.ref<i32>
! Bounds generation loads the iterator and converts it to index
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! Load i for lower bound (i)
! CHECK: %[[I_LOAD1:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_I64_1:.*]] = fir.convert %[[I_LOAD1]] : (i32) -> i64
! CHECK: %[[I_IDX_1:.*]] = fir.convert %[[I_I64_1]] : (i64) -> index
! Load i for upper bound (i+2)
! CHECK: %[[I_LOAD2:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: %[[I_PLUS_2:.*]] = arith.addi %[[I_LOAD2]], %[[C2_I32]] : i32
! CHECK: %[[UB_I64:.*]] = fir.convert %[[I_PLUS_2]] : (i32) -> i64
! CHECK: %[[UB_IDX:.*]] = fir.convert %[[UB_I64]] : (i64) -> index
! Compute lowerbound = i - startIdx (offset from startIdx)
! CHECK: %[[LB:.*]] = arith.subi %[[I_IDX_1]], %[[C1]] : index
! Compute upperbound = (i+2) - startIdx (offset from startIdx)
! CHECK: %[[UB:.*]] = arith.subi %[[UB_IDX]], %[[C1]] : index
! Compute extent = ub - lb + 1
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB]] : index
! CHECK: %[[EXT:.*]] = arith.addi %[[DIFF]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_loop_varEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_2d_loop_vars()
! Test 2D cache with swapped loop variables inside nested loop: b(j:j+1, i:i+1)
subroutine test_cache_2d_loop_vars()
  integer, parameter :: n = 10
  real, dimension(n, n) :: a, b
  integer :: i, j

  !$acc loop
  do i = 1, n-1
    do j = 1, n-1
      !$acc cache(b(j:j+1, i:i+1))
      a(i,j) = b(j,i) + b(j+1,i+1)
    end do
  end do

! CHECK: acc.loop private({{.*}}) control(%[[I_IV:.*]] : i32) = ({{.*}}) to ({{.*}})
! Outer loop iterator i is stored to privatized variable
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_2d_loop_varsEi"}
! CHECK: fir.store %[[I_IV]] to %[[I_DECL]]#0 : !fir.ref<i32>
! Inner loop j (non-acc loop, fir.do_loop)
! CHECK: fir.do_loop %[[J_IV:.*]] = {{.*}} iter_args(%[[J_ITER:.*]] = {{.*}})
! Inner loop iterator j is stored to j variable
! CHECK: fir.store %[[J_ITER]] to %[[J_REF:.*]] : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! Dimension 1 bounds from j: load j, convert, compute j+1
! CHECK: %[[J_LOAD1:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
! CHECK: %[[J_I64_1:.*]] = fir.convert %[[J_LOAD1]] : (i32) -> i64
! CHECK: %[[J_IDX_1:.*]] = fir.convert %[[J_I64_1]] : (i64) -> index
! CHECK: %[[J_LOAD2:.*]] = fir.load %[[J_REF]] : !fir.ref<i32>
! CHECK: %[[C1_I32_J:.*]] = arith.constant 1 : i32
! CHECK: %[[J_PLUS_1:.*]] = arith.addi %[[J_LOAD2]], %[[C1_I32_J]] : i32
! CHECK: %[[J_PLUS_1_I64:.*]] = fir.convert %[[J_PLUS_1]] : (i32) -> i64
! CHECK: %[[J_PLUS_1_IDX:.*]] = fir.convert %[[J_PLUS_1_I64]] : (i64) -> index
! Compute lowerbound = j - 1, upperbound = (j+1) - 1, extent = 2
! CHECK: %[[LB1:.*]] = arith.subi %[[J_IDX_1]], %[[C1]] : index
! CHECK: %[[UB1:.*]] = arith.subi %[[J_PLUS_1_IDX]], %[[C1]] : index
! CHECK: %[[DIFF1:.*]] = arith.subi %[[UB1]], %[[LB1]] : index
! CHECK: %[[EXT1:.*]] = arith.addi %[[DIFF1]], %[[C1]] : index
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) extent(%[[EXT1]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! Dimension 2 bounds from i (outer loop): load i, convert, compute i+1
! CHECK: %[[I_LOAD1:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_I64_1:.*]] = fir.convert %[[I_LOAD1]] : (i32) -> i64
! CHECK: %[[I_IDX_1:.*]] = fir.convert %[[I_I64_1]] : (i64) -> index
! CHECK: %[[I_LOAD2:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1_I32_I:.*]] = arith.constant 1 : i32
! CHECK: %[[I_PLUS_1:.*]] = arith.addi %[[I_LOAD2]], %[[C1_I32_I]] : i32
! CHECK: %[[I_PLUS_1_I64:.*]] = fir.convert %[[I_PLUS_1]] : (i32) -> i64
! CHECK: %[[I_PLUS_1_IDX:.*]] = fir.convert %[[I_PLUS_1_I64]] : (i64) -> index
! Compute lowerbound = i - 1, upperbound = (i+1) - 1, extent = 2
! CHECK: %[[LB2:.*]] = arith.subi %[[I_IDX_1]], %[[C1]] : index
! CHECK: %[[UB2:.*]] = arith.subi %[[I_PLUS_1_IDX]], %[[C1]] : index
! CHECK: %[[DIFF2:.*]] = arith.subi %[[UB2]], %[[LB2]] : index
! CHECK: %[[EXT2:.*]] = arith.addi %[[DIFF2]], %[[C1]] : index
! CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) extent(%[[EXT2]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_2d_loop_varsEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_single_element()
! Test cache with single element access: b(i)
subroutine test_cache_single_element()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(i))
    a(i) = b(i)
  end do

! CHECK: acc.loop private({{.*}}) control(%[[IV:.*]] : i32) = ({{.*}}) to ({{.*}})
! The privatized iterator is declared and initialized from the loop control variable
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_single_elementEi"}
! CHECK: fir.store %[[IV]] to %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! Load i from the iterator variable and convert to index
! CHECK: %[[I_LOAD:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_I64:.*]] = fir.convert %[[I_LOAD]] : (i32) -> i64
! CHECK: %[[I_IDX:.*]] = fir.convert %[[I_I64]] : (i64) -> index
! Compute lowerbound = i - startIdx (offset from startIdx), extent = 1 for single element
! CHECK: %[[LB:.*]] = arith.subi %[[I_IDX]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[C1]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_single_elementEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_mixed_bounds()
! Test cache with mixed constant and variable bounds: b(1:i)
subroutine test_cache_mixed_bounds()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(1:i))
    a(i) = b(i)
  end do

! CHECK: acc.loop private({{.*}}) control(%[[IV:.*]] : i32) = ({{.*}}) to ({{.*}})
! The privatized iterator is declared and initialized
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_mixed_boundsEi"}
! CHECK: fir.store %[[IV]] to %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! b(1:i): lower bound is constant 1
! CHECK: %[[C1_LB:.*]] = arith.constant 1 : index
! Upper bound i is loaded from iterator variable
! CHECK: %[[I_LOAD:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_I64:.*]] = fir.convert %[[I_LOAD]] : (i32) -> i64
! CHECK: %[[I_IDX:.*]] = fir.convert %[[I_I64]] : (i64) -> index
! Compute lowerbound = 1 - startIdx = 0 (constant offset)
! CHECK: %[[LB:.*]] = arith.subi %[[C1_LB]], %[[C1]] : index
! Compute upperbound = i - startIdx (offset from startIdx, uses iterator)
! CHECK: %[[UB:.*]] = arith.subi %[[I_IDX]], %[[C1]] : index
! Compute extent = ub - lb + 1 = i (dynamic based on iterator)
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB]] : index
! CHECK: %[[EXT:.*]] = arith.addi %[[DIFF]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_mixed_boundsEb"}
end subroutine
