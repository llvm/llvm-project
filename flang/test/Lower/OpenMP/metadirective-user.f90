! Test lowering of OpenMP metadirective with user={condition()} selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 %s -o - | FileCheck %s

!===----------------------------------------------------------------------===!
! Static (constant-folded) user conditions
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_condition_true()
! CHECK:         omp.taskyield
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_true()
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: taskyield) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_false()
! CHECK-NOT:     omp.taskwait
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_false()
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_score()
! CHECK-NOT:     omp.taskyield
! CHECK:         omp.taskwait
! CHECK:         return
subroutine test_condition_score()
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: taskyield) &
  !$omp & when(user={condition(score(2): .true.)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_condition_true()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_begin_condition_true()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(user={condition(.true.)}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(user={condition(.true.)}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_condition_false()
! CHECK-NOT:     omp.parallel
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_begin_condition_false()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(user={condition(.false.)}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(user={condition(.false.)}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

!===----------------------------------------------------------------------===!
! Dynamic (runtime) user conditions
!===----------------------------------------------------------------------===!

! CHECK-LABEL: func.func @_QPtest_dynamic_condition(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]#0
! CHECK:         %[[COND:.*]] = fir.convert %[[LOAD]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[COND]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_condition(flag)
  logical, intent(in) :: flag
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_dynamic_condition_expr(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]#0
! CHECK:         %[[C1000:.*]] = arith.constant 1000 : i32
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD]], %[[C1000]] : i32
! CHECK:         fir.if %[[CMP]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_condition_expr(n)
  integer, intent(in) :: n
  !$omp metadirective &
  !$omp & when(user={condition(n > 1000)}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! Both when clauses pass vendor(llvm) statically. The first has a dynamic
! condition so becomes a runtime branch; the second is fully static and
! becomes the fallback.
! CHECK-LABEL: func.func @_QPtest_mixed_static_dynamic(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]#0
! CHECK:         %[[C100:.*]] = arith.constant 100 : i32
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD]], %[[C100]] : i32
! CHECK:         fir.if %[[CMP]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:           omp.taskwait
! CHECK:         }
! CHECK:         return
subroutine test_mixed_static_dynamic(n)
  integer, intent(in) :: n
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}, user={condition(n > 100)}: barrier) &
  !$omp & when(implementation={vendor(llvm)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! Dynamic candidate whose static traits don't match is skipped entirely.
! CHECK-LABEL: func.func @_QPtest_dynamic_static_mismatch(
! CHECK-NOT:     fir.if
! CHECK:         omp.taskyield
! CHECK:         return
subroutine test_dynamic_static_mismatch(flag)
  logical, intent(in) :: flag
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}, user={condition(flag)}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(taskyield)
#else
  !$omp & default(taskyield)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_two_dynamic(
! CHECK-SAME:    %[[ARG0:[^,]*]]: !fir.ref<!fir.logical<4>>
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DECLA:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[DECLB:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[LOADA:.*]] = fir.load %[[DECLA]]#0
! CHECK:         %[[CONDA:.*]] = fir.convert %[[LOADA]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[CONDA]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:           %[[LOADB:.*]] = fir.load %[[DECLB]]#0
! CHECK:           %[[CONDB:.*]] = fir.convert %[[LOADB]] : (!fir.logical<4>) -> i1
! CHECK:           fir.if %[[CONDB]] {
! CHECK:             omp.taskwait
! CHECK:           } else {
! CHECK:             omp.taskyield
! CHECK:           }
! CHECK:         }
! CHECK:         return
subroutine test_two_dynamic(a, b)
  logical, intent(in) :: a, b
  !$omp metadirective &
  !$omp & when(user={condition(a)}: barrier) &
  !$omp & when(user={condition(b)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(taskyield)
#else
  !$omp & default(taskyield)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_three_dynamic(
! CHECK-SAME:    %[[A:[^,]*]]: !fir.ref<!fir.logical<4>>
! CHECK-SAME:    %[[B:[^,]*]]: !fir.ref<!fir.logical<4>>
! CHECK-SAME:    %[[C:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DA:.*]]:2 = hlfir.declare %[[A]]
! CHECK:         %[[DB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:         %[[DC:.*]]:2 = hlfir.declare %[[C]]
! CHECK:         %[[LA:.*]] = fir.load %[[DA]]#0
! CHECK:         %[[CA:.*]] = fir.convert %[[LA]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[CA]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:           %[[LB:.*]] = fir.load %[[DB]]#0
! CHECK:           %[[CB:.*]] = fir.convert %[[LB]] : (!fir.logical<4>) -> i1
! CHECK:           fir.if %[[CB]] {
! CHECK:             omp.taskwait
! CHECK:           } else {
! CHECK:             %[[LC:.*]] = fir.load %[[DC]]#0
! CHECK:             %[[CC:.*]] = fir.convert %[[LC]] : (!fir.logical<4>) -> i1
! CHECK:             fir.if %[[CC]] {
! CHECK:               omp.taskyield
! CHECK:             } else {
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:         return
subroutine test_three_dynamic(a, b, c)
  logical, intent(in) :: a, b, c
  !$omp metadirective &
  !$omp & when(user={condition(a)}: barrier) &
  !$omp & when(user={condition(b)}: taskwait) &
  !$omp & when(user={condition(c)}: taskyield) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_multi_dynamic_multi_static(
! CHECK-SAME:    %[[A:[^,]*]]: !fir.ref<!fir.logical<4>>
! CHECK-SAME:    %[[B:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DA:.*]]:2 = hlfir.declare %[[A]]
! CHECK:         %[[DB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:         %[[LA:.*]] = fir.load %[[DA]]#0
! CHECK:         %[[CA:.*]] = fir.convert %[[LA]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[CA]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:           %[[LB:.*]] = fir.load %[[DB]]#0
! CHECK:           %[[CB:.*]] = fir.convert %[[LB]] : (!fir.logical<4>) -> i1
! CHECK:           fir.if %[[CB]] {
! CHECK:             omp.taskyield
! CHECK:           } else {
! CHECK:             omp.taskwait
! CHECK:           }
! CHECK:         }
! CHECK:         return
subroutine test_multi_dynamic_multi_static(a, b)
  logical, intent(in) :: a, b
  ! dynamic + vendor(llvm) -> kept
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}, user={condition(a)}: barrier) &
  ! dynamic + vendor("unknown") -> skipped (static mismatch)
  !$omp & when(implementation={vendor("unknown")}, user={condition(.true.)}: taskwait) &
  ! dynamic + vendor(llvm) -> kept
  !$omp & when(implementation={vendor(llvm)}, user={condition(b)}: taskyield) &
  ! static + vendor(llvm) -> best static fallback
  !$omp & when(implementation={vendor(llvm)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine
