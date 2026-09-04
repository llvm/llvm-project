! Test lowering of imperfectly nested collapse loops (CLN relaxation).
! Intervening code is guarded by IV comparisons to restore correct
! execution frequency and ordering within the flat omp.loop_nest body.

! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcollapse2_imperfect
subroutine collapse2_imperfect(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    x = x + 1
    do j = 1, n
      x = x + j
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{.*}}, %[[LB_J:.*]]) to
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Guard: j == lower_bound (before code executes once per i)
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LB_J]] : i32
! CHECK:           fir.if %[[CMP]] {
! Intervening code: x = x + 1
! CHECK:             %[[X1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[C1:.*]] = arith.constant 1 : i32
! CHECK:             %[[ADD1:.*]] = arith.addi %[[X1]], %[[C1]] : i32
! CHECK:             hlfir.assign %[[ADD1]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           }
! Innermost body: x = x + j
! CHECK:           %[[X2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[JVAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[ADD2:.*]] = arith.addi %[[X2]], %[[JVAL]] : i32
! CHECK:           hlfir.assign %[[ADD2]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           omp.yield

! CHECK-LABEL: func.func @_QPcollapse3_imperfect
subroutine collapse3_imperfect(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j, k

  !$omp do collapse(3)
  do i = 1, n
    x = x + i
    do j = 1, n
      x = x + j
      do k = 1, n
        x = x + k
      end do
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I3:.*]], %[[J3:.*]], %[[K3:.*]]) : i32 =
! CHECK-SAME:      (%{{.*}}, %[[LB_J3:.*]], %[[LB_K3:.*]]) to
! CHECK:           hlfir.assign %[[I3]]
! CHECK:           hlfir.assign %[[J3]]
! CHECK:           hlfir.assign %[[K3]]
! Guard: j == lb_j AND k == lb_k (level 0 before code, once per i)
! CHECK:           %[[CMP_J:.*]] = arith.cmpi eq, %[[J3]], %[[LB_J3]] : i32
! CHECK:           %[[CMP_K1:.*]] = arith.cmpi eq, %[[K3]], %[[LB_K3]] : i32
! CHECK:           %[[AND1:.*]] = arith.andi %[[CMP_J]], %[[CMP_K1]] : i1
! CHECK:           fir.if %[[AND1]] {
! Intervening code at level 0: x = x + i
! CHECK:             %[[XI:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[IVAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[ADDI:.*]] = arith.addi %[[XI]], %[[IVAL]] : i32
! CHECK:             hlfir.assign %[[ADDI]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           }
! Guard: k == lb_k (level 1 before code, once per (i,j))
! CHECK:           %[[CMP_K2:.*]] = arith.cmpi eq, %[[K3]], %[[LB_K3]] : i32
! CHECK:           fir.if %[[CMP_K2]] {
! Intervening code at level 1: x = x + j
! CHECK:             %[[XJ:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[JVAL3:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[ADDJ:.*]] = arith.addi %[[XJ]], %[[JVAL3]] : i32
! CHECK:             hlfir.assign %[[ADDJ]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           }
! Innermost body: x = x + k
! CHECK:           %[[XK:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[KVAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[ADDK:.*]] = arith.addi %[[XK]], %[[KVAL]] : i32
! CHECK:           hlfir.assign %[[ADDK]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           omp.yield

! CHECK-LABEL: func.func @_QPcollapse2_both_sides
subroutine collapse2_both_sides(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp simd collapse(2)
  do i = 1, n
    x = x + 1
    do j = 1, n
      x = x + j
    end do
    call ext_sub(x)
  end do
  !$omp end simd
end subroutine

! CHECK:       omp.simd
! CHECK-NEXT:    omp.loop_nest (%[[I4:.*]], %[[J4:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J4:[^)]*]]) to (%{{[^)]*}}, %[[UB_J4:[^)]*]])
! CHECK:           hlfir.assign %[[I4]]
! CHECK:           hlfir.assign %[[J4]]
! Guard: j == lower_bound (before code)
! CHECK:           %[[CMP_B:.*]] = arith.cmpi eq, %[[J4]], %[[LB_J4]] : i32
! CHECK:           fir.if %[[CMP_B]] {
! Intervening code before inner loop: x = x + 1
! CHECK:             %[[XB:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[CB:.*]] = arith.constant 1 : i32
! CHECK:             %[[ADDB:.*]] = arith.addi %[[XB]], %[[CB]] : i32
! CHECK:             hlfir.assign %[[ADDB]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:           }
! Innermost body: x = x + j
! CHECK:           %[[XIN:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[JIN:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[ADDIN:.*]] = arith.addi %[[XIN]], %[[JIN]] : i32
! CHECK:           hlfir.assign %[[ADDIN]] to %{{.*}} : i32, !fir.ref<i32>
! Guard: j == upper_bound (after code)
! CHECK:           %[[CMP_A:.*]] = arith.cmpi eq, %[[J4]], %[[UB_J4]] : i32
! CHECK:           fir.if %[[CMP_A]] {
! Intervening code after inner loop: call ext_sub(x)
! CHECK:             fir.call @_QPext_sub
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(3) with both before and after code at multiple levels.
! CHECK-LABEL: func.func @_QPcollapse3_both_sides
subroutine collapse3_both_sides(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j, k

  !$omp do collapse(3)
  do i = 1, n
    x = x + i
    do j = 1, n
      x = x + j
      do k = 1, n
        x = x + k
      end do
      x = x - j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]], %[[K:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]], %[[LB_K:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]], %[[UB_K:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
!
! --- "before" guards (outermost level first) ---
!
! Guard level 0 before: j == lb_j AND k == lb_k (once per i)
! CHECK:           %[[CJ1:.*]] = arith.cmpi eq, %[[J]], %[[LB_J]] : i32
! CHECK:           %[[CK1:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           %[[AND1:.*]] = arith.andi %[[CJ1]], %[[CK1]] : i1
! CHECK:           fir.if %[[AND1]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Guard level 1 before: k == lb_k (once per (i,j))
! CHECK:           %[[CK2:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           fir.if %[[CK2]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
!
! --- innermost body: x = x + k ---
! CHECK:           arith.addi
! CHECK:           hlfir.assign
!
! --- "after" guards (innermost level first) ---
!
! Guard level 1 after: k == ub_k (once per (i,j))
! CHECK:           %[[CK3:.*]] = arith.cmpi eq, %[[K]], %[[UB_K]] : i32
! CHECK:           fir.if %[[CK3]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! Guard level 0 after: j == ub_j AND k == ub_k (once per i)
! CHECK:           %[[CJ2:.*]] = arith.cmpi eq, %[[J]], %[[UB_J]] : i32
! CHECK:           %[[CK4:.*]] = arith.cmpi eq, %[[K]], %[[UB_K]] : i32
! CHECK:           %[[AND2:.*]] = arith.andi %[[CJ2]], %[[CK4]] : i1
! CHECK:           fir.if %[[AND2]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(4) with imperfect nesting at some levels and perfectly nested
! innermost loops. Level 0 (i->j) has before+after, level 1 (j->k) has before
! only, level 2 (k->l) is perfectly nested. This exercises skipping empty levels.
! CHECK-LABEL: func.func @_QPcollapse4_mixed
subroutine collapse4_mixed(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j, k, l

  !$omp do collapse(4)
  do i = 1, n
    x = x + i
    do j = 1, n
      x = x + j
      do k = 1, n
        do l = 1, n
          x = x + l
        end do
      end do
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]], %[[K:.*]], %[[L:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]], %[[LB_K:[^)]*]], %[[LB_L:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]], %[[UB_K:[^)]*]], %[[UB_L:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! CHECK:           hlfir.assign %[[L]]
!
! --- "before" guards ---
!
! Guard level 0 before: j == lb_j AND k == lb_k AND l == lb_l (once per i)
! CHECK:           %[[CJ1:.*]] = arith.cmpi eq, %[[J]], %[[LB_J]] : i32
! CHECK:           %[[CK1:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           %[[A1:.*]] = arith.andi %[[CJ1]], %[[CK1]] : i1
! CHECK:           %[[CL1:.*]] = arith.cmpi eq, %[[L]], %[[LB_L]] : i32
! CHECK:           %[[A2:.*]] = arith.andi %[[A1]], %[[CL1]] : i1
! CHECK:           fir.if %[[A2]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Guard level 1 before: k == lb_k AND l == lb_l (once per (i,j))
! CHECK:           %[[CK2:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           %[[CL2:.*]] = arith.cmpi eq, %[[L]], %[[LB_L]] : i32
! CHECK:           %[[A3:.*]] = arith.andi %[[CK2]], %[[CL2]] : i1
! CHECK:           fir.if %[[A3]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Level 2 (k->l) is perfectly nested: no guard emitted.
!
! --- innermost body: x = x + l ---
! CHECK:           arith.addi
! CHECK:           hlfir.assign
!
! --- "after" guards (innermost first) ---
!
! Level 2 after: empty (perfectly nested), no guard emitted.
! Level 1 after: empty, no guard emitted.
! Guard level 0 after: j == ub_j AND k == ub_k AND l == ub_l (once per i)
! CHECK:           %[[CJ2:.*]] = arith.cmpi eq, %[[J]], %[[UB_J]] : i32
! CHECK:           %[[CK3:.*]] = arith.cmpi eq, %[[K]], %[[UB_K]] : i32
! CHECK:           %[[A4:.*]] = arith.andi %[[CJ2]], %[[CK3]] : i1
! CHECK:           %[[CL3:.*]] = arith.cmpi eq, %[[L]], %[[UB_L]] : i32
! CHECK:           %[[A5:.*]] = arith.andi %[[A4]], %[[CL3]] : i1
! CHECK:           fir.if %[[A5]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(2) with only after-code (no before-code). Exercises the path
! where levels[i].before.empty() is true and the "before" loop is entirely skipped.
! CHECK-LABEL: func.func @_QPcollapse2_after_only
subroutine collapse2_after_only(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 1, n
      x = x + j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %{{[^)]*}}) to (%{{[^)]*}}, %[[UB_J:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! No "before" guard emitted (level 0 before is empty).
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Guard: j == upper_bound (after code)
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[UB_J]] : i32
! CHECK:           fir.if %[[CMP]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(2) with multiple statements inside a single guard. Verifies
! that all evals in level.before land inside the same fir.if block.
! CHECK-LABEL: func.func @_QPcollapse2_multi_stmt
subroutine collapse2_multi_stmt(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    x = x + 1
    x = x + i
    do j = 1, n
      x = x + j
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]]) to
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Guard: j == lower_bound (before code, multiple statements in one guard)
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LB_J]] : i32
! CHECK:           fir.if %[[CMP]] {
! First intervening statement: x = x + 1
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! Second intervening statement: x = x + i
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(2) with non-unit lower bound on inner loop. Verifies the guard
! compares against the actual loop lower bound operand (3, not 1).
! CHECK-LABEL: func.func @_QPcollapse2_nonunit_lb
subroutine collapse2_nonunit_lb(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    x = x + i
    do j = 3, n
      x = x + j
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]]) to
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Guard: j == lb_j (lb_j is 3, not 1)
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LB_J]] : i32
! CHECK:           fir.if %[[CMP]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(3) with after-only at level 0 and before-only at level 1.
! Exercises the independent skip logic at each level in both emission loops.
! CHECK-LABEL: func.func @_QPcollapse3_mixed_sides
subroutine collapse3_mixed_sides(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j, k

  !$omp do collapse(3)
  do i = 1, n
    do j = 1, n
      x = x + j
      do k = 1, n
        x = x + k
      end do
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]], %[[K:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %{{[^)]*}}, %[[LB_K:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]], %[[UB_K:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! Level 0 before: empty (skipped).
! Guard level 1 before: k == lb_k (once per (i,j))
! CHECK:           %[[CK:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           fir.if %[[CK]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Innermost body: x = x + k
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 1 after: empty (skipped).
! Guard level 0 after: j == ub_j AND k == ub_k (once per i)
! CHECK:           %[[CJ:.*]] = arith.cmpi eq, %[[J]], %[[UB_J]] : i32
! CHECK:           %[[CK2:.*]] = arith.cmpi eq, %[[K]], %[[UB_K]] : i32
! CHECK:           %[[AND:.*]] = arith.andi %[[CJ]], %[[CK2]] : i1
! CHECK:           fir.if %[[AND]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(2) with non-unit positive step and after-code.
! The after guard must compare iv against the last executed value
! (lb + ((ub - lb) / step) * step), not the upper bound directly.
! For do j = 1, 10, 4: last_iv = 1 + ((10-1)/4)*4 = 1 + 8 = 9.
! CHECK-LABEL: func.func @_QPcollapse2_nonunit_step_after
subroutine collapse2_nonunit_step_after(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 1, 10, 4
      x = x + j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]]) inclusive step (%{{[^)]*}}, %[[STEP_J:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After guard: compute last_iv = lb + ((ub - lb) / step) * step
! CHECK:           %[[RANGE:.*]] = arith.subi %[[UB_J]], %[[LB_J]] : i32
! CHECK:           %[[DIV:.*]] = arith.divsi %[[RANGE]], %[[STEP_J]] : i32
! CHECK:           %[[MUL:.*]] = arith.muli %[[DIV]], %[[STEP_J]] : i32
! CHECK:           %[[LAST:.*]] = arith.addi %[[LB_J]], %[[MUL]] : i32
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LAST]] : i32
! CHECK:           fir.if %[[CMP]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(2) with negative step and after-code.
! For do j = 10, 1, -4: last_iv = 10 + ((1-10)/(-4))*(-4) = 10 + (2*-4) = 2.
! CHECK-LABEL: func.func @_QPcollapse2_negative_step_after
subroutine collapse2_negative_step_after(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 10, 1, -4
      x = x + j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]]) inclusive step (%{{[^)]*}}, %[[STEP_J:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After guard: compute last_iv for negative step
! CHECK:           %[[RANGE:.*]] = arith.subi %[[UB_J]], %[[LB_J]] : i32
! CHECK:           %[[DIV:.*]] = arith.divsi %[[RANGE]], %[[STEP_J]] : i32
! CHECK:           %[[MUL:.*]] = arith.muli %[[DIV]], %[[STEP_J]] : i32
! CHECK:           %[[LAST:.*]] = arith.addi %[[LB_J]], %[[MUL]] : i32
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LAST]] : i32
! CHECK:           fir.if %[[CMP]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(3) with non-unit step on the middle loop (not innermost).
! For do j = 1, n, 3: last_iv = lb + ((ub - lb) / step) * step (runtime).
! CHECK-LABEL: func.func @_QPcollapse3_nonunit_step_middle
subroutine collapse3_nonunit_step_middle(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j, k

  !$omp do collapse(3)
  do i = 1, n
    do j = 1, n, 3
      x = x + j
      do k = 1, n
        x = x + k
      end do
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]], %[[K:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]], %[[LB_K:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]], %[[UB_K:[^)]*]]) inclusive step (%{{[^)]*}}, %[[STEP_J:[^)]*]], %{{[^)]*}})
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! Guard level 1 before: k == lb_k (once per (i,j))
! CHECK:           %[[CK1:.*]] = arith.cmpi eq, %[[K]], %[[LB_K]] : i32
! CHECK:           fir.if %[[CK1]] {
! CHECK:             arith.addi
! CHECK:             hlfir.assign
! CHECK:           }
! Innermost body: x = x + k
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Guard level 0 after: must compute last_iv for j (non-unit step) AND k == ub_k
! CHECK:           %[[RANGE:.*]] = arith.subi %[[UB_J]], %[[LB_J]] : i32
! CHECK:           %[[DIV:.*]] = arith.divsi %[[RANGE]], %[[STEP_J]] : i32
! CHECK:           %[[MUL:.*]] = arith.muli %[[DIV]], %[[STEP_J]] : i32
! CHECK:           %[[LASTJ:.*]] = arith.addi %[[LB_J]], %[[MUL]] : i32
! CHECK:           %[[CJ:.*]] = arith.cmpi eq, %[[J]], %[[LASTJ]] : i32
! CHECK:           %[[CK2:.*]] = arith.cmpi eq, %[[K]], %[[UB_K]] : i32
! CHECK:           %[[AND:.*]] = arith.andi %[[CJ]], %[[CK2]] : i1
! CHECK:           fir.if %[[AND]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test collapse(2) with a dynamic (runtime) step value.
! The step is not a compile-time constant, so the last_iv computation
! cannot be folded away and must remain as arith ops in the IR.
! CHECK-LABEL: func.func @_QPcollapse2_dynamic_step_after
subroutine collapse2_dynamic_step_after(n, s, x)
  integer, intent(in) :: n, s
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 1, n, s
      x = x + j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %[[LB_J:[^)]*]]) to (%{{[^)]*}}, %[[UB_J:[^)]*]]) inclusive step (%{{[^)]*}}, %[[STEP_J:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After guard: dynamic step forces last_iv computation to stay in IR
! CHECK:           %[[RANGE:.*]] = arith.subi %[[UB_J]], %[[LB_J]] : i32
! CHECK:           %[[DIV:.*]] = arith.divsi %[[RANGE]], %[[STEP_J]] : i32
! CHECK:           %[[MUL:.*]] = arith.muli %[[DIV]], %[[STEP_J]] : i32
! CHECK:           %[[LAST:.*]] = arith.addi %[[LB_J]], %[[MUL]] : i32
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[LAST]] : i32
! CHECK:           fir.if %[[CMP]] {
! CHECK:             arith.subi
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield

! Test that "after" code reading an inner DO variable sees the Fortran
! terminal value (lb + tripcount*step, i.e. one past the last executed
! value), not the last executed value the flattened nest leaves it at.
! The after guard must restore j to ub + step before the after code runs.
! CHECK-LABEL: func.func @_QPcollapse2_after_reads_inner
subroutine collapse2_after_reads_inner(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 1, n
      x = x + 1
    end do
    x = x + j
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK-SAME:      (%{{[^)]*}}, %{{[^)]*}}) to (%{{[^)]*}}, %[[UB_J:[^)]*]]) inclusive step (%{{[^)]*}}, %[[STEP_J:[^)]*]])
! CHECK:           hlfir.assign %[[I]]
! Capture j's storage from the loop-variable store so the restore and the
! after-code read can be tied to the same address.
! CHECK:           hlfir.assign %[[J]] to %[[J_ADDR:.*]] : i32, !fir.ref<i32>
! Innermost body: x = x + 1
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Guard: j == ub_j (unit-step fast path compares against the upper bound)
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[J]], %[[UB_J]] : i32
! CHECK:           fir.if %[[CMP]] {
! Restore j's storage to its Fortran terminal value (ub + step).
! CHECK:             %[[TERM:.*]] = arith.addi %[[UB_J]], %[[STEP_J]] : i32
! CHECK:             hlfir.assign %[[TERM]] to %[[J_ADDR]] : i32, !fir.ref<i32>
! After code reads the restored j: x = x + j.
! CHECK:             %[[XLD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:             %[[JLD:.*]] = fir.load %[[J_ADDR]] : !fir.ref<i32>
! CHECK:             arith.addi %[[XLD]], %[[JLD]] : i32
! CHECK:             hlfir.assign
! CHECK:           }
! CHECK:           omp.yield
