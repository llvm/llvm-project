! Test lowering of metadirectives with ordinary loop-associated variants.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! CHECK: #loop_unroll = #llvm.loop_unroll<disable = false, count = 4 : i64>
! CHECK: #loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>

! CHECK-LABEL: func.func @_QPtest_do(
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:             omp.yield
! CHECK:         return
subroutine test_do(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! The score is compared before explicitness, so a higher-scored implicit
! NOTHING is selected over an explicit DO when its condition is true.
! CHECK-LABEL: func.func @_QPtest_implicit_nothing_score(
! CHECK:         %[[FLAG:.*]] = fir.load {{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:         %[[COND:.*]] = fir.convert %[[FLAG]]
! CHECK:         fir.if %[[COND]] {
! CHECK-NOT:       omp.
! CHECK:           fir.do_loop
! CHECK:         } else {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:         }
! CHECK:         return
subroutine test_implicit_nothing_score(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(10): flag)}:) &
  !$omp & when(user={condition(score(5): .true.)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_simd(
! CHECK-NOT:     omp.wsloop
! CHECK:         omp.simd linear(
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:             omp.yield
! CHECK-NOT:     fir.do_loop
! CHECK:         fir.load
! CHECK:         hlfir.assign
! CHECK:         return
subroutine test_simd(n, a, after)
  integer :: n, a(n), after, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
end subroutine

! CHECK-LABEL: func.func @_QPtest_do_simd(
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.simd
! CHECK:             omp.loop_nest
! CHECK:               hlfir.assign
! CHECK:               omp.yield
! CHECK:         return
subroutine test_do_simd(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_do(
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:             omp.yield
! CHECK:         return
subroutine test_begin_do(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! The following loop must remain available when the PFT is reused for ENTRY.
! CHECK-LABEL: func.func @_QPtest_standalone_entry_no_directive(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:         } else {
! CHECK:           fir.do_loop
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK:         return
! CHECK-LABEL: func.func @_QPtest_alt_standalone_entry_no_directive(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:         } else {
! CHECK:           fir.do_loop
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK:         return
! CHECK-LABEL: func.func @_QPtest_after_standalone_entry_no_directive(
! CHECK-NOT:     fir.if
! CHECK-NOT:     omp.
! CHECK-NOT:     fir.do_loop
! CHECK:         %[[AFTER_ENTRY_C77:.*]] = arith.constant 77 : i32
! CHECK:         hlfir.assign %[[AFTER_ENTRY_C77]]
! CHECK:         return
subroutine test_standalone_entry_no_directive(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  entry test_alt_standalone_entry_no_directive(flag, n, a)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  entry test_after_standalone_entry_no_directive(n, a)
  a(1) = 77
end subroutine

! Intervening compiler directives have the same ownership across ENTRY.
! CHECK-LABEL: func.func @_QPtest_standalone_entry(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:         } else {
! CHECK:           fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK:         return
! CHECK-LABEL: func.func @_QPtest_alt_standalone_entry(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:         } else {
! CHECK:           fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK:         return
subroutine test_standalone_entry(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  entry test_alt_standalone_entry(flag, n, a)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  !dir$ unroll 4
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A statically inapplicable loop variant leaves the following loop after the
! selected standalone variant, so it is lowered sequentially.
! CHECK-LABEL: func.func @_QPtest_static_standalone_fallback(
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         omp.barrier
! CHECK:         fir.do_loop
! CHECK:           hlfir.assign
! CHECK-NOT:     fir.do_loop
! CHECK:         return
subroutine test_static_standalone_fallback(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: do) &
  !$omp & otherwise(barrier)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A statically inapplicable loop variant nested in a parallel region leaves the
! following loop sequential.
! CHECK-LABEL: func.func @_QPtest_inapplicable_do_in_parallel(
! CHECK:         omp.parallel
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:           fir.do_loop
! CHECK:             hlfir.assign
! CHECK-NOT:       fir.do_loop
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_inapplicable_do_in_parallel(n, a, after)
  integer :: n, a(n), after, i
  !$omp parallel num_threads(1) shared(n, a, after)
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

! A lower-ranked loop variant is unreachable after an unguarded standalone
! variant is selected, so it does not impose loop-only lowering restrictions.
! CHECK-LABEL: func.func @_QPtest_unselected_do_in_parallel(
! CHECK:         omp.parallel
! CHECK:           omp.barrier
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:           fir.do_loop
! CHECK:             hlfir.assign
! CHECK-NOT:       fir.do_loop
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_unselected_do_in_parallel(n, a)
  integer :: n, a(n), i
  !$omp parallel num_threads(1) shared(n, a)
  !$omp metadirective &
  !$omp & when(user={condition(score(2): .true.)}: barrier) &
  !$omp & when(user={condition(score(1): .true.)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end parallel
end subroutine

! An unreachable loop variant likewise does not turn a statically selected
! block variant into a mixed-association metadirective.
! CHECK-LABEL: func.func @_QPtest_unselected_do_with_block_variant(
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         omp.masked
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:           fir.do_loop
! CHECK:             hlfir.assign
! CHECK-NOT:       fir.do_loop
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_unselected_do_with_block_variant(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(score(2): .true.)}: masked) &
  !$omp & when(user={condition(score(1): .true.)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! A lower-ranked candidate guarded by the same runtime expression is
! unreachable: when FLAG is true the higher-ranked BARRIER wins, and when it
! is false neither guarded candidate matches. Do not emit a dead OpenMP loop.
! CHECK-LABEL: func.func @_QPtest_unreachable_same_runtime_condition(
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:         } else {
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK:           hlfir.assign
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         return
subroutine test_unreachable_same_runtime_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Parentheses do not make a repeatable condition distinct. The lower-ranked
! loop remains unreachable and must not be emitted.
! CHECK-LABEL: func.func @_QPtest_unreachable_parenthesized_condition(
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         return
subroutine test_unreachable_parenthesized_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): (flag))}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Idempotent AND/OR spelling is normalized after proving that the condition is
! repeatable.
! CHECK-LABEL: func.func @_QPtest_unreachable_idempotent_condition(
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK-NOT:       omp.wsloop
! CHECK-NOT:       omp.loop_nest
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK-NOT:     omp.wsloop
! CHECK-NOT:     omp.loop_nest
! CHECK:         return
subroutine test_unreachable_idempotent_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): flag .or. flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Calls to an opaque procedure are independent runtime conditions even when
! their source expressions are identical. Preserve both candidates without
! relying on clause-expression side effects.
! CHECK-LABEL: func.func @_QPtest_opaque_runtime_conditions(
! CHECK:         fir.call @_QPmetadirective_runtime_condition
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:           fir.call @_QPmetadirective_runtime_condition
! CHECK:           fir.if {{.*}} {
! CHECK:             omp.wsloop
! CHECK:               omp.loop_nest
! CHECK:           } else {
! CHECK:             fir.do_loop
! CHECK:           }
! CHECK:         }
! CHECK:         return
subroutine test_opaque_runtime_conditions(n, a)
  integer :: n, a(n), i
  logical :: metadirective_runtime_condition
  external :: metadirective_runtime_condition
  !$omp metadirective &
  !$omp & when(user={condition(score(2): metadirective_runtime_condition())}: barrier) &
  !$omp & when(user={condition(score(1): metadirective_runtime_condition())}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

module metadirective_condition_helpers
contains
  pure logical function metadirective_identity(value)
    logical, intent(in) :: value
    metadirective_identity = value
  end function
end module

! Procedure calls are conservatively kept as independent runtime conditions
! because the expression tree does not describe the callee's state.
! CHECK-LABEL: func.func @_QPtest_pure_runtime_conditions(
! CHECK:         fir.call @_QMmetadirective_condition_helpersPmetadirective_identity
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:         } else {
! CHECK:           fir.call @_QMmetadirective_condition_helpersPmetadirective_identity
! CHECK:           fir.if {{.*}} {
! CHECK:             omp.simd
! CHECK:               omp.loop_nest
! CHECK:           } else {
! CHECK:             fir.do_loop
! CHECK:           }
! CHECK:         }
! CHECK:         return
subroutine test_pure_runtime_conditions(flag, n, a)
  use metadirective_condition_helpers, only : metadirective_identity
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): &
  !$omp &   metadirective_identity(flag))}: do) &
  !$omp & when(user={condition(score(1): &
  !$omp &   metadirective_identity(flag))}: simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_dynamic_loop(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:               hlfir.assign
! CHECK:         } else {
! CHECK:           omp.simd
! CHECK:             omp.loop_nest
! CHECK:               hlfir.assign
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_loop(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(simd)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! When the standalone fallback is selected at runtime, the following loop is
! lowered sequentially in that arm.
! CHECK-LABEL: func.func @_QPtest_dynamic_standalone_fallback(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:               hlfir.assign
! CHECK:         } else {
! CHECK:           omp.barrier
! CHECK:           fir.do_loop
! CHECK:             hlfir.assign
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_standalone_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(barrier)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! When NOTHING is selected, the following loop is lowered normally.
! CHECK-LABEL: func.func @_QPtest_dynamic_nothing_fallback(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:               hlfir.assign
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK:           fir.do_loop
! CHECK:             hlfir.assign
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_nothing_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Compiler directives preceding the associated loop are processed before it.
! CHECK-LABEL: func.func @_QPtest_dynamic_unroll_fallback(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.wsloop
! CHECK:         } else {
! CHECK:           fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_unroll_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  !dir$ unroll 4
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Each runtime arm must compute its own affected depth and restore temporary
! loop-index attributes before lowering the next arm.
! CHECK-LABEL: func.func @_QPtest_dynamic_collapse(
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.simd {{.*}}private({{.*}}Ei_private_i32{{.*}}Ej_private_i32
! CHECK:             omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2)
! CHECK:               hlfir.assign
! CHECK:         } else {
! CHECK:           omp.simd linear(
! CHECK:             omp.loop_nest ({{.*}}) : i32
! CHECK:               fir.do_loop
! CHECK:                 hlfir.assign
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_collapse(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: simd collapse(2)) &
  !$omp & otherwise(simd)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_ordered_depth(
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop {{.*}}private({{.*}}Ei_private_i32{{.*}}Ej_private_i32
! CHECK:           omp.loop_nest ({{.*}}) : i32
! CHECK:             fir.do_loop
! CHECK:               hlfir.assign
! CHECK:         return
subroutine test_ordered_depth(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do ordered(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_schedule(
! CHECK:         omp.wsloop schedule(static)
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:         return
subroutine test_schedule(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do schedule(static)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2)
! CHECK:             hlfir.assign
! CHECK:         return
subroutine test_collapse(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do collapse(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_safelen(
! CHECK:         omp.simd {{.*}}safelen(4)
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:         return
subroutine test_safelen(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd safelen(4)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! SIMD collapse makes every affected index lastprivate in OpenMP 5.2. Check
! that lowering copies both private values back to their source bindings.
! CHECK-LABEL: func.func @_QPtest_simd_collapse_lastprivate(
! CHECK:         %[[I:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_simd_collapse_lastprivateEi"}
! CHECK:         %[[J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_simd_collapse_lastprivateEj"}
! CHECK:         omp.simd {{.*}}private({{.*}}Ei_private_i32{{.*}}Ej_private_i32
! CHECK:           omp.loop_nest ({{.*}}, {{.*}}) : i32 {{.*}} collapse(2)
! CHECK:             fir.if
! CHECK:               hlfir.assign {{.*}} to %[[I]]#0
! CHECK:               hlfir.assign {{.*}} to %[[J]]#0
! CHECK:             omp.yield
! CHECK:         return
subroutine test_simd_collapse_lastprivate(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd collapse(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_block_nested_do(
! CHECK-NOT:     omp.parallel
! CHECK:         omp.wsloop {{.*}}private({{.*}}Ei_private_i32
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:         return
subroutine test_block_nested_do(n, a)
  integer :: n, a(n), i
  block
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: do) &
    !$omp & otherwise(nothing)
    do i = 1, n
      a(i) = i
    end do
  end block
end subroutine

! A selected block variant owns sequential-loop IVs in its body.
! CHECK-LABEL: func.func @_QPtest_block_owned_iv(
! CHECK:         omp.parallel {{.*}}private({{.*}}Ei_private_i32
! CHECK:           %[[I:.*]]:2 = hlfir.declare
! CHECK:           fir.do_loop
! CHECK:             fir.store %{{.*}} to %[[I]]#0
! CHECK:             %{{.*}} = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:         return
subroutine test_block_owned_iv(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! A nested SIMD construct owns its associated loop IV. The selected outer
! PARALLEL must not replace the IV's predetermined LINEAR attribute with
! PRIVATE, but it must still privatize a sequential loop nested in the SIMD
! region beyond its associated depth.
! CHECK-LABEL: func.func @_QPtest_block_nested_simd_iv(
! CHECK:         omp.parallel {
! CHECK:           %[[J:.*]] = fir.alloca i32
! CHECK:           hlfir.declare %[[J]]
! CHECK:           omp.simd linear(val(
! CHECK:             omp.loop_nest
! CHECK:               fir.do_loop
! CHECK:         return
subroutine test_block_nested_simd_iv(n, a)
  integer :: n, a(n, n), i, j
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel) &
  !$omp& otherwise(nothing)
  !$omp simd
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
  !$omp end simd
  !$omp end metadirective
end subroutine

! A selected TASK applies its implicit FIRSTPRIVATE to the SIMD IV's outer
! association while preserving LINEAR on the nested construct-local symbol.
! CHECK-LABEL: func.func @_QPtest_task_nested_simd_iv(
! CHECK:         omp.task {{.*}}private({{.*}}Ei_firstprivate_i32
! CHECK:           omp.simd linear(val(
! CHECK:             omp.loop_nest
! CHECK:         return
subroutine test_task_nested_simd_iv(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task) &
  !$omp& otherwise(nothing)
  !$omp simd
  do i = 1, n
    a(i) = i
  end do
  !$omp end simd
  !$omp end metadirective
end subroutine

! A selected block variant must not privatize an enclosing worksharing loop's
! predetermined iteration variable.
! CHECK-LABEL: func.func @_QPtest_enclosing_do_iv(
! CHECK:         omp.wsloop {{.*}}private({{.*}}Ei_private_i32
! CHECK:           omp.loop_nest
! CHECK:             %[[I:.*]]:2 = hlfir.declare
! CHECK:             omp.parallel {
! CHECK:               %{{.*}} = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:         return
subroutine test_enclosing_do_iv(n, a)
  integer :: n, a(n), i
  !$omp do
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel) &
    !$omp& otherwise(nothing)
    a(i) = i
    !$omp end metadirective
  end do
end subroutine

! A predetermined flag left on a sibling transform's IV must not make a
! selected SINGLE try to privatize that symbol.
! CHECK-LABEL: func.func @_QPtest_sibling_transform_single(
! CHECK:         %[[I:.*]]:2 = hlfir.declare {{.*}}Ei"
! CHECK:         omp.single {
! CHECK:           %{{.*}} = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:         return
subroutine test_sibling_transform_single(n, a)
  integer :: n, a(n), i
  !$omp unroll partial(2)
  do i = 1, n
    a(i) = i
  end do
  !$omp begin metadirective &
  !$omp& when(user={condition(.true.)}: single) &
  !$omp& otherwise(nothing)
  a(1) = i
  !$omp end metadirective
end subroutine

! A selected TASK must apply its own implicit firstprivate rule instead of
! inheriting a predetermined flag from a sibling transform.
! CHECK-LABEL: func.func @_QPtest_sibling_transform_task(
! CHECK:         omp.task {{.*}}private({{.*}}Ei_firstprivate_i32
! CHECK:           %[[I:.*]]:2 = hlfir.declare {{.*}}Ei"
! CHECK:           %{{.*}} = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:         return
subroutine test_sibling_transform_task(n, a)
  integer :: n, a(n), i
  !$omp unroll partial(2)
  do i = 1, n
    a(i) = i
  end do
  !$omp begin metadirective &
  !$omp& when(user={condition(.true.)}: task) &
  !$omp& otherwise(nothing)
  a(1) = i
  !$omp end metadirective
end subroutine

! A standalone metadirective's selected DO trait remains active while its
! sibling loop is traversed. The inner construct selector therefore chooses
! NOTHING, leaving the inner loop sequential.
! CHECK-LABEL: func.func @_QPtest_standalone_selected_construct_context(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK-NOT:         omp.wsloop
! CHECK:             fir.do_loop
! CHECK-NOT:           omp.wsloop
! CHECK:               hlfir.assign
! CHECK-NOT:     omp.wsloop
! CHECK:         return
subroutine test_standalone_selected_construct_context(n, a)
  integer :: n, a(n), i, j
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={do}: nothing) &
    !$omp& otherwise(do)
    do j = 1, 2
      a(i) = j
    end do
  end do
end subroutine

! An intervening parallel region breaks close nesting, so its barrier remains
! valid inside a selected DO replacement.
! CHECK-LABEL: func.func @_QPtest_barrier_nested_in_parallel(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             omp.parallel
! CHECK:               omp.barrier
! CHECK:               omp.terminator
! CHECK:             omp.yield
! CHECK:         return
subroutine test_barrier_nested_in_parallel(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp parallel shared(a) firstprivate(i)
    !$omp barrier
    a(i) = i
    !$omp end parallel
  end do
end subroutine

! A PARALLEL replacement selected by a nested metadirective also breaks close
! nesting. Validate the BARRIER against the realized replacement rather than
! rejecting the nested metadirective conservatively.
! CHECK-LABEL: func.func @_QPtest_barrier_nested_in_metadirective_parallel(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             omp.parallel
! CHECK:               omp.barrier
! CHECK:               omp.terminator
! CHECK:             omp.yield
! CHECK:         return
subroutine test_barrier_nested_in_metadirective_parallel(n)
  integer :: n, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel) &
    !$omp& otherwise(nothing)
    !$omp barrier
    !$omp end metadirective
  end do
end subroutine

! An unreachable nested BARRIER replacement does not constrain the selected
! outer loop.
! CHECK-LABEL: func.func @_QPtest_unreachable_nested_barrier(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK-NOT:         omp.barrier
! CHECK:             omp.yield
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_unreachable_nested_barrier(n)
  integer :: n, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={parallel}: barrier) &
    !$omp& otherwise(nothing)
  end do
end subroutine
