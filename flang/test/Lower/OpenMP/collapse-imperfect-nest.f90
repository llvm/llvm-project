! Test lowering of imperfectly nested collapse loops (CLN relaxation).
! Intervening code is emitted unguarded into the flat omp.loop_nest body, so it
! executes once per collapsed logical iteration. OpenMP 6.0 leaves the count
! unspecified between once per enclosing iteration and once per logical
! iteration.

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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Intervening code: x = x + 1
! CHECK:           %[[X1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[C1:.*]] = arith.constant 1 : i32
! CHECK:           %[[ADD1:.*]] = arith.addi %[[X1]], %[[C1]] : i32
! CHECK:           hlfir.assign %[[ADD1]] to %[[XADDR:.*]] : i32, !fir.ref<i32>
! Innermost body: x = x + j
! CHECK:           %[[X2:.*]] = fir.load %[[XADDR]] : !fir.ref<i32>
! CHECK:           %[[JVAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[ADD2:.*]] = arith.addi %[[X2]], %[[JVAL]] : i32
! CHECK:           hlfir.assign %[[ADD2]] to %[[XADDR]] : i32, !fir.ref<i32>
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
! CHECK:           hlfir.assign %[[I3]]
! CHECK:           hlfir.assign %[[J3]]
! CHECK:           hlfir.assign %[[K3]]
! Level 0 intervening code: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 1 intervening code: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + k
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Intervening code on both sides of the inner loop, emitted in source order.
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
! CHECK:           hlfir.assign %[[I4]]
! CHECK:           hlfir.assign %[[J4]]
! Before the inner loop: x = x + 1
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After the inner loop: call ext_sub(x)
! CHECK:           fir.call @_QPext_sub
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

! Emission order is: before code outermost-to-innermost, innermost body, then
! after code innermost-to-outermost.
! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]], %[[K:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! Level 0 before: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 1 before: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + k
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 1 after: x = x - j
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! Level 0 after: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! CHECK:           hlfir.assign %[[L]]
! Level 0 before: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 1 before: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + l
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 0 after: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(2) with only after-code (no before-code). Exercises the path
! where levels[i].before is empty.
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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After code: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(2) with multiple intervening statements at one level.
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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! First intervening statement: x = x + 1
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Second intervening statement: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(2) with a non-unit lower bound and a runtime step on the inner
! loop.
! CHECK-LABEL: func.func @_QPcollapse2_runtime_step
subroutine collapse2_runtime_step(n, s, x)
  integer, intent(in) :: n, s
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    x = x + i
    do j = 3, n, s
      x = x + j
    end do
    x = x - i
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Before code: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After code: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Test collapse(3) with after-only at level 0 and before-only at level 1.
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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! CHECK:           hlfir.assign %[[K]]
! Level 1 before: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + k
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Level 0 after: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! OpenMP 6.0 6.4.3 requires each collapsed loop's iteration variable to hold the
! value it would have in the unassociated nest, so "after" code reads j from the
! current logical iteration rather than a synthesized Fortran terminal value.
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
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]] to %[[J_ADDR:.*]] : i32, !fir.ref<i32>
! Innermost body: x = x + 1
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After code reads j directly, with no restore of a terminal value.
! CHECK:           %[[XLD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[JLD:.*]] = fir.load %[[J_ADDR]] : !fir.ref<i32>
! CHECK:           arith.addi %[[XLD]], %[[JLD]] : i32
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Labeled DO form: the terminating CONTINUE survives canonicalization as a
! sibling of the inner loop.
! CHECK-LABEL: func.func @_QPcollapse2_labeled_do
subroutine collapse2_labeled_do(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do 10 i = 1, n
    x = x + i
    do 20 j = 1, n
      x = x + j
20  continue
    x = x - i
10 continue
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Before code: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! After code: x = x - i
! CHECK:           arith.subi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! A compiler directive between the loops is transparent to perfect nesting.
! CHECK-LABEL: func.func @_QPcollapse2_compiler_directive
subroutine collapse2_compiler_directive(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    x = x + i
    !dir$ vector always
    do j = 1, n
      x = x + j
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Before code: x = x + i
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Intervening code need not be straight-line: a block IF is valid intervening
! code and brings its own control flow into the loop_nest body.
! CHECK-LABEL: func.func @_QPcollapse2_if_construct
subroutine collapse2_if_construct(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(2)
  do i = 1, n
    if (i > 2) then
      x = x + i
    end if
    do j = 1, n
      x = x + j
    end do
  end do
  !$omp end do
end subroutine

! CHECK:       omp.wsloop
! CHECK-NEXT:    omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]]
! Before code: if (i > 2) x = x + i
! CHECK:           arith.cmpi sgt
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Innermost body: x = x + j
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield

! Intervening code inside a target region, whose collapsed bounds are host_eval
! block arguments of the enclosing omp.target.
! CHECK-LABEL: func.func @_QPcollapse2_target_intervening
subroutine collapse2_target_intervening(n, m, x)
  integer, intent(in) :: n, m
  integer, intent(inout) :: x
  integer :: i, j

  !$omp target teams distribute parallel do collapse(2) map(tofrom:x)
  do i = 1, n
    do j = 1, m
      x = x + 1
    end do
    x = x + j
  end do
end subroutine

! CHECK:       omp.target
! CHECK-SAME:    host_eval(
! CHECK:         omp.loop_nest (%[[I:.*]], %[[J:.*]]) : i32 =
! CHECK:           hlfir.assign %[[I]]
! CHECK:           hlfir.assign %[[J]] to %[[J_ADDR:.*]] : i32, !fir.ref<i32>
! Innermost body: x = x + 1
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! Intervening code: x = x + j
! CHECK:           %[[JLD:.*]] = fir.load %[[J_ADDR]] : !fir.ref<i32>
! CHECK:           arith.addi
! CHECK:           hlfir.assign
! CHECK:           omp.yield
