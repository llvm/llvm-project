! Test lowering of OpenMP metadirective with loop-associated variants.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck %s

!===----------------------------------------------------------------------===!
! Basic loop-associated variants via static selection
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_parallel_do()
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:             omp.yield
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_parallel_do()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_do()
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:           omp.yield
! CHECK:         return
subroutine test_do()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_simd()
! CHECK:         %[[I:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_simdEi"}
! CHECK-NOT:     omp.wsloop
! CHECK:         omp.simd linear(%[[I]]#0
! CHECK:           omp.loop_nest
! CHECK:           omp.yield
! CHECK-NOT:     fir.do_loop
! CHECK:         fir.load %[[I]]#0
! CHECK:         return
subroutine test_simd()
  integer :: i, i_after
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd) &
  !$omp & default(nothing)
  do i = 1, 100
    i_after = i
  end do
  i_after = i
end subroutine

! CHECK-LABEL: func.func @_QPtest_do_simd()
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.simd
! CHECK:             omp.loop_nest
! CHECK:             omp.yield
! CHECK:         return
subroutine test_do_simd()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do simd) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_do()
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:           omp.yield
! CHECK:         return
subroutine test_begin_do()
  integer :: i
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
  !$omp end metadirective
end subroutine

!===----------------------------------------------------------------------===!
! Static mismatch falls through to standalone fallback
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_loop_static_mismatch()
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_loop_static_mismatch()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: parallel do) &
  !$omp & default(barrier)
  do i = 1, 100
  end do
end subroutine

!===----------------------------------------------------------------------===!
! Dynamic user condition with loop-associated variant
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_dynamic_loop(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]#0
! CHECK:         %[[COND:.*]] = fir.convert %[[LOAD]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[COND]] {
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK:               omp.loop_nest
! CHECK:         } else {
! CHECK:           omp.simd
! CHECK:             omp.loop_nest
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_loop(flag)
  logical, intent(in) :: flag
  integer :: i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: parallel do) &
  !$omp & default(simd)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_dynamic_loop_standalone_fallback(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK:               omp.loop_nest
! CHECK:         } else {
! CHECK:           omp.barrier
! CHECK:           fir.do_loop
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_loop_standalone_fallback(flag, a)
  logical, intent(in) :: flag
  integer :: i, a
  a = 0
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: parallel do) &
  !$omp & default(barrier)
  do i = 1, 100
    a = a + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_dynamic_loop_dsa_isolation(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.simd {{.*}}private(@{{[^,]*}}Ei_private_i32
! CHECK-SAME:      @_QFtest_dynamic_loop_dsa_isolationEj_private_i32
! CHECK:             omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2)
! CHECK:         } else {
! CHECK:           omp.simd linear({{[^)]*}}) {
! CHECK:             omp.loop_nest ({{.*}}) : i32
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_loop_dsa_isolation(flag, n, sink)
  logical, intent(in) :: flag
  integer :: n, sink
  integer :: i, j
  sink = 0
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: simd collapse(2)) &
  !$omp & default(simd)
  do i = 1, n
    do j = 1, n
      sink = sink + i + j
    end do
  end do
  sink = sink + j
end subroutine

! CHECK-LABEL: func.func @_QPtest_ordered_depth()
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop {{.*}}private(@{{[^,]*}}Ei_private_i32
! CHECK-SAME:      @_QFtest_ordered_depthEj_private_i32
! CHECK:             omp.loop_nest ({{.*}}) : i32
! CHECK:         return
subroutine test_ordered_depth()
  integer :: i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do ordered(2)) &
  !$omp & default(nothing)
  do i = 1, 100
    do j = 1, 100
    end do
  end do
end subroutine

!===----------------------------------------------------------------------===!
! Loop-associated variants with clauses
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_schedule()
! CHECK:         omp.wsloop schedule(static)
! CHECK:           omp.loop_nest
! CHECK:         return
subroutine test_schedule()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do schedule(static)) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_reduction()
! CHECK:         omp.wsloop {{.*}} reduction(@add_reduction_i32
! CHECK:           omp.loop_nest
! CHECK:         return
subroutine test_reduction()
  integer :: i, s
  s = 0
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do reduction(+:s)) &
  !$omp & default(nothing)
  do i = 1, 100
    s = s + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private()
! CHECK:         omp.wsloop private(
! CHECK:           omp.loop_nest
! CHECK:         return
subroutine test_private()
  integer :: i, x
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do private(x)) &
  !$omp & default(nothing)
  do i = 1, 100
    x = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse()
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2)
! CHECK:         return
subroutine test_collapse()
  integer :: i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do collapse(2)) &
  !$omp & default(nothing)
  do i = 1, 100
    do j = 1, 100
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_safelen()
! CHECK:         omp.simd {{.*}}safelen(4)
! CHECK:           omp.loop_nest
! CHECK:         return
subroutine test_safelen()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd safelen(4)) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_num_threads()
! CHECK:         omp.parallel num_threads({{.*}}) {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:         return
subroutine test_num_threads()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do num_threads(4)) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine

!===----------------------------------------------------------------------===!
! Lastprivate copy-back for metadirective-selected loop induction variables
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_simd_collapse_lastprivate(
! CHECK:         %[[I_ORIG:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_simd_collapse_lastprivateEi"}
! CHECK:         %[[J_ORIG:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_simd_collapse_lastprivateEj"}
! CHECK:         omp.simd {{.*}}private({{.*}}_QFtest_simd_collapse_lastprivateEi_private_i32{{.*}}_QFtest_simd_collapse_lastprivateEj_private_i32{{.*}}) {
! CHECK:           omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2) {
! CHECK:             fir.if
! CHECK:               hlfir.assign %{{.*}} to %[[I_ORIG]]#0
! CHECK:               hlfir.assign %{{.*}} to %[[J_ORIG]]#0
! CHECK:             omp.yield
subroutine test_simd_collapse_lastprivate(n, sink)
  integer :: n, sink
  integer :: i, j
  sink = 0
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd collapse(2)) &
  !$omp & default(nothing)
  do i = 1, n
    do j = 1, n
      sink = sink + i + j
    end do
  end do
  sink = sink + j
end subroutine

!===----------------------------------------------------------------------===!
! Metadirective nested inside a BLOCK construct
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_block_nested_parallel_do(
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop private(@_QFtest_block_nested_parallel_doEi_private_i32 {{.*}}) {
! CHECK:             omp.loop_nest
subroutine test_block_nested_parallel_do(n, a)
  integer :: n
  integer :: a(n)
  integer :: i
  block
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: parallel do) &
    !$omp & default(nothing)
    do i = 1, n
      a(i) = i
    end do
  end block
end subroutine

!===----------------------------------------------------------------------===!
! Inner sequential loop induction-variable privatization
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_parallel_do_seq_inner(
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop private({{.*}}_QFtest_parallel_do_seq_innerEi_private_i32{{.*}}_QFtest_parallel_do_seq_innerEk_private_i32{{.*}}) {
! CHECK:             omp.loop_nest
subroutine test_parallel_do_seq_inner(n, a)
  integer :: n
  integer :: a(n, n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, n
    do k = 1, n
      a(k, i) = k
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_do_seq_inner_shared(
! CHECK:         omp.wsloop private(
! CHECK-NOT:       _QFtest_do_seq_inner_sharedEk_private
! CHECK:           omp.loop_nest
subroutine test_do_seq_inner_shared(n, a)
  integer :: n
  integer :: a(n, n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & default(nothing)
  do i = 1, n
    do k = 1, n
      a(k, i) = k
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_do_critical_inner(
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop private({{.*}}_QFtest_parallel_do_critical_innerEi_private_i32{{.*}}_QFtest_parallel_do_critical_innerEk_private_i32{{.*}}) {
! CHECK:             omp.loop_nest
! CHECK:               omp.critical
subroutine test_parallel_do_critical_inner(n, a)
  integer :: n
  integer :: a(n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, n
    !$omp critical
    do k = 1, n
      a(k) = k
    end do
    !$omp end critical
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_do_inner_shared(
! CHECK:         omp.wsloop private(
! CHECK:           omp.loop_nest
! CHECK:             omp.parallel {
! CHECK-NOT:           _QFtest_parallel_do_inner_sharedEk_private
subroutine test_parallel_do_inner_shared(n, a)
  integer :: n
  integer :: a(n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, n
    !$omp parallel shared(k)
    do k = 1, n
      a(k) = k
    end do
    !$omp end parallel
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_do_reduction_iv(
! CHECK:         omp.wsloop private(@_QFtest_parallel_do_reduction_ivEi_private_i32 {{[^,]*}}) reduction(@add_reduction_i32 {{.*}}) {
subroutine test_parallel_do_reduction_iv(n, a)
  integer :: n
  integer :: a(n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do reduction(+:k)) &
  !$omp & default(nothing)
  do i = 1, n
    do k = 1, n
      a(k) = k
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_do_enclosing_shared(
! CHECK:         omp.wsloop private({{.*}}_QFtest_parallel_do_enclosing_sharedEi_private_i32{{.*}}_QFtest_parallel_do_enclosing_sharedEk_private_i32{{.*}}) {
! CHECK:           omp.loop_nest
subroutine test_parallel_do_enclosing_shared(n, a)
  integer :: n
  integer :: a(n)
  integer :: i, k
  !$omp parallel shared(k)
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, n
    do k = 1, n
      a(k) = k
    end do
  end do
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_do_target_inner(
! CHECK:         omp.wsloop private(@_QFtest_parallel_do_target_innerEi_private_i32 {{[^,]*}}) {
! CHECK:           omp.target
! CHECK-NOT:         _QFtest_parallel_do_target_innerEk_private_i32
subroutine test_parallel_do_target_inner(n, a)
  integer :: n
  integer :: a(n)
  integer :: i, k
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, n
    !$omp target map(tofrom: a)
    do k = 1, n
      a(k) = k
    end do
    !$omp end target
  end do
end subroutine
