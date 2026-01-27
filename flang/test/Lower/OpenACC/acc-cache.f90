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
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_basicEb"}
! Loop body uses the cached reference
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[DECL]]#0 (%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[ELEM]] : !fir.ref<f32>
! CHECK: hlfir.assign %[[LOAD]] to %{{.*}} : f32, !fir.ref<f32>
! Scope termination: acc.yield marks the end of the cache scope
! CHECK: acc.yield
! CHECK-NEXT: }
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
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {modifiers = #acc<data_clause_modifier readonly>, name = "b"
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_readonlyEb"}
! Loop body uses the cached readonly reference
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[DECL]]#0 (%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[ELEM]] : !fir.ref<f32>
! CHECK: hlfir.assign %[[LOAD]] to %{{.*}} : f32, !fir.ref<f32>
! Scope termination
! CHECK: acc.yield
! CHECK-NEXT: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_array_section()
! For b(2:5) with startIdx=1: lowerbound = 2-1 = 1, upperbound = 5-1 = 4, extent = 4
! This test includes an IF statement to verify cache scope with unstructured control flow
subroutine test_cache_array_section()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(2:5))
    if (i > 2) then
      a(i) = b(i)
    end if
  end do

! CHECK: acc.loop
! For b(2:5) with startIdx=1: lowerbound = 2-1 = 1, upperbound = 5-1 = 4
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 1 : index
! CHECK: %[[UB:.*]] = arith.constant 4 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_array_sectionEb"}
! Unstructured control flow: IF condition generates fir.if
! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
! CHECK: fir.if %[[CMP]] {
! Loop body uses the cached array section inside conditional
! CHECK:   hlfir.designate %[[DECL]]#0
! CHECK:   fir.load
! CHECK:   hlfir.assign
! CHECK: }
! Scope termination: acc.yield terminates the cache scope for all control flow paths
! CHECK: acc.yield
! CHECK-NEXT: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_multiple()
! This test includes IF-ELSE to verify cache scope with multiple control flow paths
subroutine test_cache_multiple()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b, c)
    if (i < 5) then
      a(i) = b(i) + c(i)
    else
      a(i) = b(i) - c(i)
    end if
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE_B:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b"
! CHECK: %[[DECL_B:.*]]:2 = hlfir.declare %[[CACHE_B]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEb"}
! CHECK: %[[CACHE_C:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "c"
! CHECK: %[[DECL_C:.*]]:2 = hlfir.declare %[[CACHE_C]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEc"}
! Unstructured control flow: IF-ELSE generates fir.if with else region
! CHECK: %[[CMP:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
! CHECK: fir.if %[[CMP]] {
! Then branch: uses both cached references with addition
! CHECK:   hlfir.designate %[[DECL_B]]#0
! CHECK:   fir.load
! CHECK:   hlfir.designate %[[DECL_C]]#0
! CHECK:   fir.load
! CHECK:   arith.addf
! CHECK:   hlfir.assign
! CHECK: } else {
! Else branch: uses both cached references with subtraction
! CHECK:   hlfir.designate %[[DECL_B]]#0
! CHECK:   fir.load
! CHECK:   hlfir.designate %[[DECL_C]]#0
! CHECK:   fir.load
! CHECK:   arith.subf
! CHECK:   hlfir.assign
! CHECK: }
! Scope termination: both IF and ELSE paths use cache, then converge to yield
! CHECK: acc.yield
! CHECK-NEXT: }
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
! Dimension 1: lowerbound = 1 - startIdx = 0, upperbound = 5 - startIdx = 4
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[C4:.*]] = arith.constant 4 : index
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[C4]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! Dimension 2: lowerbound = 0, upperbound = 4
! CHECK: %[[C0_2:.*]] = arith.constant 0 : index
! CHECK: %[[C4_2:.*]] = arith.constant 4 : index
! CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[C0_2]] : index) upperbound(%[[C4_2]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_2d_arrayEb"}
! Nested loop uses the cached 2D array
! CHECK: fir.do_loop
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: hlfir.assign
! Scope termination for acc.loop
! CHECK: acc.yield
! CHECK-NEXT: }
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
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! Load iterator i for lowerbound computation
! CHECK: %[[I_LOAD1:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_CVT1:.*]] = fir.convert %[[I_LOAD1]] : (i32) -> i64
! CHECK: %[[I_IDX1:.*]] = fir.convert %[[I_CVT1]] : (i64) -> index
! Compute lowerbound = i - 1
! CHECK: %[[LB:.*]] = arith.subi %[[I_IDX1]], %[[C1]] : index
! Load iterator i again for upperbound computation (i+2)
! CHECK: %[[I_LOAD2:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C2:.*]] = arith.constant 2 : i32
! CHECK: %[[I_PLUS_2:.*]] = arith.addi %[[I_LOAD2]], %[[C2]] : i32
! CHECK: %[[UB_CVT:.*]] = fir.convert %[[I_PLUS_2]] : (i32) -> i64
! CHECK: %[[UB_IDX:.*]] = fir.convert %[[UB_CVT]] : (i64) -> index
! Compute upperbound = (i+2) - 1
! CHECK: %[[UB:.*]] = arith.subi %[[UB_IDX]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_loop_varEb"}
! Loop body uses the cached reference for b(i), b(i+1), b(i+2)
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: arith.addf
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: arith.addf
! CHECK: hlfir.assign
! Scope termination
! CHECK: acc.yield
! CHECK-NEXT: }
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
! Dimension 1 bounds from j: lowerbound = j-1, upperbound = j
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! Dimension 2 bounds from i: lowerbound = i-1, upperbound = i
! CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_2d_loop_varsEb"}
! Loop body uses the cached 2D reference
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: arith.addf
! CHECK: hlfir.assign
! Inner loop continues within the cache scope
! CHECK: }
! Scope termination for acc.loop
! CHECK: acc.yield
! CHECK-NEXT: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_single_element()
! Test cache with single element access: b(i)
! This test includes an EXIT statement to verify cache scope with early loop exit
subroutine test_cache_single_element()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(i))
    a(i) = b(i)
    if (a(i) > 100.0) exit
  end do

! Unstructured loop with EXIT: acc.loop becomes unstructured with cf.br/cf.cond_br
! CHECK: acc.loop private({{.*}}) {
! The privatized iterator is declared
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_single_elementEi"}
! Loop control is done with cf.br/cf.cond_br in unstructured form
! CHECK: cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK: cf.cond_br %{{.*}}, ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! Load iterator i for bounds computation
! CHECK: %[[I_LOAD:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_CVT1:.*]] = fir.convert %[[I_LOAD]] : (i32) -> i64
! CHECK: %[[I_IDX:.*]] = fir.convert %[[I_CVT1]] : (i64) -> index
! Compute lowerbound = i - 1 (single element: upperbound = lowerbound)
! CHECK: %[[LB:.*]] = arith.subi %[[I_IDX]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[LB]] : index) extent(%[[C1]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_single_elementEb"}
! Loop body uses the cached single element
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: fir.load
! CHECK: hlfir.assign
! Unstructured control flow: EXIT generates conditional branch
! CHECK: %[[CMP:.*]] = arith.cmpf ogt, %{{.*}}, %{{.*}} : f32
! CHECK: cf.cond_br %[[CMP]], ^[[EXIT_BB:.*]], ^[[CONT_BB:.*]]
! CHECK: ^[[EXIT_BB]]:
! Early exit path: branch to acc.yield
! CHECK: cf.br ^[[YIELD:.*]]
! CHECK: ^[[CONT_BB]]:
! Normal path: update iterator and loop back
! CHECK: cf.br ^[[HEADER]]
! CHECK: ^[[YIELD]]:
! Scope termination: acc.yield marks end of cache scope
! CHECK: acc.yield
! CHECK-NEXT: } attributes {{{.*}}unstructured}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_mixed_bounds()
! Test cache with mixed constant and variable bounds: b(1:i)
! This test includes a CYCLE statement to verify cache scope with loop continuation
subroutine test_cache_mixed_bounds()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(1:i))
    if (mod(i, 2) == 0) cycle
    a(i) = b(i)
  end do

! CHECK: acc.loop private({{.*}}) control(%[[IV:.*]] : i32) = ({{.*}}) to ({{.*}})
! The privatized iterator is declared and initialized
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_cache_mixed_boundsEi"}
! CHECK: fir.store %[[IV]] to %[[I_DECL]]#0 : !fir.ref<i32>
! b(1:i): lower bound is constant 0 (1-1), upper bound is i-1
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! Load iterator i for upperbound computation
! CHECK: %[[I_LOAD:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[I_CVT:.*]] = fir.convert %[[I_LOAD]] : (i32) -> i64
! CHECK: %[[I_IDX:.*]] = fir.convert %[[I_CVT]] : (i64) -> index
! Compute upperbound = i - 1
! CHECK: %[[UB:.*]] = arith.subi %[[I_IDX]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_mixed_boundsEb"}
! Unstructured control flow: CYCLE generates inverted fir.if (body executes when NOT cycling)
! CHECK: %[[MOD:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
! CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[MOD]], %{{.*}} : i32
! CHECK: %[[TRUE:.*]] = arith.constant true
! CHECK: %[[NOT_CYCLE:.*]] = arith.xori %[[CMP]], %[[TRUE]] : i1
! CHECK: fir.if %[[NOT_CYCLE]] {
! Loop body uses the cached reference (only executed when not cycling)
! CHECK:   hlfir.designate %[[DECL]]#0
! CHECK:   fir.load
! CHECK:   hlfir.assign
! CHECK: }
! Scope termination: acc.yield after the conditional
! CHECK: acc.yield
! CHECK-NEXT: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_nonunit_lb()
! Test cache with array that has non-1 lower bound: arr(10:20), cache(arr(15))
! This test includes SELECT CASE for multi-way unstructured control flow
subroutine test_cache_nonunit_lb()
  integer :: arr(10:20)
  integer :: i

  !$acc loop
  do i = 10, 20
    !$acc cache(arr(15))
    select case (mod(i, 3))
    case (0)
      arr(i) = i * 2
    case (1)
      arr(i) = i * 3
    case default
      arr(i) = i
    end select
  end do

! For arr(10:20), startIdx = 10, element 15 has lowerbound = 15 - 10 = 5
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! Unstructured loop with SELECT CASE: acc.loop becomes unstructured
! CHECK: acc.loop private({{.*}}) {
! CHECK: cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK: cf.cond_br %{{.*}}, ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! Compute lowerbound = 15 - startIdx = 15 - 10 = 5
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C15:.*]] = arith.constant 15 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C15]], %{{.*}} : index
! Single element: upperbound equals lowerbound, startIdx = 10
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[LB]] : index) extent(%[[C1]] : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! For non-unit lower bound arrays, acc.cache uses the box type from hlfir.declare
! CHECK: %[[CACHE:.*]] = acc.cache var(%{{.*}} : !fir.box<!fir.array<11xi32>>) bounds(%[[BOUND]]) -> !fir.box<!fir.array<11xi32>> {{{.*}}name = "arr
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_nonunit_lbEarr"}
! Unstructured control flow: SELECT CASE generates fir.select_case
! CHECK: %[[MOD:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
! CHECK: fir.select_case %[[MOD]] : i32 [#fir.point, %{{.*}}, ^[[CASE0:.*]], #fir.point, %{{.*}}, ^[[CASE1:.*]], unit, ^[[DEFAULT:.*]]]
! Case 0: i * 2
! CHECK: ^[[CASE0]]:
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: hlfir.assign
! CHECK: cf.br ^[[MERGE:.*]]
! Case 1: i * 3
! CHECK: ^[[CASE1]]:
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: hlfir.assign
! CHECK: cf.br ^[[MERGE]]
! Default case: i
! CHECK: ^[[DEFAULT]]:
! CHECK: hlfir.designate %[[DECL]]#0
! CHECK: hlfir.assign
! CHECK: cf.br ^[[MERGE]]
! All SELECT CASE branches converge, then loop back or exit
! CHECK: ^[[MERGE]]:
! CHECK: cf.br ^[[HEADER]]
! CHECK: ^[[EXIT]]:
! Scope termination: acc.yield marks end of cache scope
! CHECK: acc.yield
! CHECK-NEXT: } attributes {{{.*}}unstructured}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_use_after_region()
! CHECK: %[[B_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_cache_use_after_regionEb"}
subroutine test_cache_use_after_region()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  end do

  ! Use b after the cache region - should use original variable
  a(1) = b(1) + 1.0

! CHECK: acc.loop
! CHECK: acc.cache varPtr(%[[B_VAR]]#0 : !fir.ref<!fir.array<10xf32>>)
! CHECK: acc.yield
! CHECK: }
! After loop: uses original b, not cached version
! CHECK: %[[B_ORIG:.*]] = hlfir.designate %[[B_VAR]]#0 (%{{.*}})
! CHECK: fir.load %[[B_ORIG]]
! CHECK: arith.addf
! CHECK: hlfir.assign
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_nested_scope()
! CHECK: %[[B_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_cache_nested_scopeEb"}
subroutine test_cache_nested_scope()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c
  integer :: i, j

  !$acc loop
  do i = 1, n
    !$acc loop
    do j = 1, n
      !$acc cache(b(j))
      c(j) = b(j)
    end do
    ! Use b(i) in outer loop - should use original, not inner cache
    a(i) = b(i)
  end do

! Outer acc.loop
! CHECK: acc.loop
! Inner acc.loop with cache
! CHECK: acc.loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%[[B_VAR]]#0 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}})
! CHECK: %[[CACHE_DECL:.*]]:2 = hlfir.declare %[[CACHE]]
! CHECK: hlfir.designate %[[CACHE_DECL]]#0
! CHECK: acc.yield
! CHECK: }
! After inner loop: uses original b, not cached
! CHECK: hlfir.designate %[[B_VAR]]#0
! CHECK: acc.yield
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_in_regular_loop()
! CHECK: %[[B_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_cache_in_regular_loopEb"}
subroutine test_cache_in_regular_loop()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  ! Cache in regular DO loop (not acc loop)
  do i = 1, n
    !$acc cache(b(i))
    a(i) = b(i)
  end do

! CHECK: fir.do_loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%[[B_VAR]]#0 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}})
! CHECK: %[[CACHE_DECL:.*]]:2 = hlfir.declare %[[CACHE]]
! CHECK: hlfir.designate %[[CACHE_DECL]]#0
! CHECK: fir.load
! CHECK: hlfir.assign
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_in_if
! CHECK: %[[B_VAR:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_cache_in_ifEb"}
subroutine test_cache_in_if(a, b, cache)
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  logical :: cache
  integer :: i

  !$acc loop
  do i = 1, n
    if (cache) then
      !$acc cache(b)
    end if
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: fir.if
! CHECK: acc.cache varPtr(%[[B_VAR]]#0 : !fir.ref<!fir.array<10xf32>>)
! CHECK: }
! After IF: uses original b, not cached version
! CHECK: hlfir.designate %[[B_VAR]]#0
! CHECK: acc.yield
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_in_nested_do
! CHECK: %[[B_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_cache_in_nested_doEb"}
subroutine test_cache_in_nested_do()
  integer, parameter :: n = 1000, m = 100, l = 100
  real, dimension(n, m, l) :: a, b
  integer :: i, j

  !$acc loop
  do i = 1, n
    do j = 1, m, 2
      !$acc cache(b(i,m,j))
    end do

    do j = 1, m, 2
      a(i, m, :) = b(i, m, :)
    end do
  end do

! CHECK: acc.loop
! First inner DO loop with cache
! CHECK: fir.do_loop
! CHECK: acc.cache varPtr(%[[B_VAR]]#0 : !fir.ref<!fir.array<1000x100x100xf32>>) bounds
! Second inner DO loop: uses original b, not cached version
! CHECK: fir.do_loop
! CHECK: hlfir.designate %[[B_VAR]]#0
end subroutine
