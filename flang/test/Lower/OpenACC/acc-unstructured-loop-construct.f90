! Each sub-file exercises a different unstructured-CFG pattern inside an
! `acc loop` whose default parallelism resolves to `independent`.
!
! In each, the branching is confined to the loop body, so the loop keeps its
! structured form and carries the construct that owns it along: the directive's
! own acc.loop holds the bounds. That holds either way
! --emit-independent-loops-as-unstructured is set.

! RUN: split-file %s %t

! RUN: bbc -fopenacc -emit-hlfir %t/goto_one_level.f90 -o - | FileCheck %s --check-prefix=GOTO1
! RUN: bbc -fopenacc -emit-hlfir %t/goto_with_intermediate.f90 -o - | FileCheck %s --check-prefix=GOTO2
! RUN: bbc -fopenacc -emit-hlfir %t/collapse_cycle.f90 -o - | FileCheck %s --check-prefix=CCYCLE
! RUN: bbc -fopenacc -emit-hlfir %t/cache_select_case.f90 -o - | FileCheck %s --check-prefix=CCASE

! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/goto_one_level.f90 -o - | FileCheck %s --check-prefix=GOTO1
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/goto_with_intermediate.f90 -o - | FileCheck %s --check-prefix=GOTO2
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/collapse_cycle.f90 -o - | FileCheck %s --check-prefix=CCYCLE
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/cache_select_case.f90 -o - | FileCheck %s --check-prefix=CCASE

!--- goto_one_level.f90

! GOTO exits the inner `acc loop seq` (one level), landing in the body of
! the outer `acc loop gang vector`. Outer loop defaults to `independent`.
subroutine test_unstructured6(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
20  B(i) = A(i)
100 continue
end subroutine

! The outer loop the directive owns: bounds on the op, and its body, which the
! GOTO branches within, folded into a region.
! GOTO1-LABEL: func.func @_QPtest_unstructured6
! GOTO1:         acc.loop gang vector private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! GOTO1:         scf.execute_region no_inline {
!
! The inner loop, which the GOTO leaves: no bounds on the op, raw branching,
! and marked unstructured.
! GOTO1:             acc.loop private({{.*}}) {
! GOTO1:               cf.cond_br
! GOTO1:               acc.yield
! GOTO1:             } seq  unstructured
!
! Nothing further is nested in the outer loop, and it ends structured.
! GOTO1-NOT:         acc.loop
! GOTO1:           acc.yield
! GOTO1-NEXT:    } inclusiveUpperbound({{.*}}) independent

!--- goto_with_intermediate.f90

! Same as above but with intermediate code between the inner loop end and
! the GOTO target, exercising the jump-table dispatch path.
subroutine test_unstructured7(A, B, C, N)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*), C(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
    C(i) = 999.0
20  B(i) = A(i)
100 continue
end subroutine

! Same nest as goto_one_level: the outer loop keeps its bounds and wraps its
! body, the inner one the GOTO leaves keeps neither.
! GOTO2-LABEL: func.func @_QPtest_unstructured7
! GOTO2:         acc.loop gang vector private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! GOTO2:         scf.execute_region no_inline {
! GOTO2:             acc.loop private({{.*}}) {
! GOTO2:               acc.yield
! GOTO2:             } seq  unstructured
!
! The jump table: the selector the inner loop stored decides whether the
! intermediate code between the loop end and the GOTO target runs.
! GOTO2:             %[[SEL:.*]] = fir.load
! GOTO2:             arith.cmpi eq, %[[SEL]], %{{.*}} : i32
! GOTO2-NEXT:        cf.cond_br
!
! GOTO2-NOT:         acc.loop
! GOTO2:           acc.yield
! GOTO2-NEXT:    } inclusiveUpperbound({{.*}}) independent

!--- collapse_cycle.f90

! Orphan `acc loop collapse(2)` with an early-exit (CYCLE) - defaults to
! `independent` inside the (non-seq) acc routine.
subroutine test_unstructured_collapse_loop_only(a)
  !$acc routine gang
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc loop collapse(2)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
end subroutine

! One loop for both collapsed levels: two induction variables on the op, and
! the body the CYCLE branches within folded into a region. A second acc.loop
! anywhere would mean a level landed on a loop of its own.
! CCYCLE-LABEL: func.func @_QPtest_unstructured_collapse_loop_only
! CCYCLE:         acc.loop private({{.*}}) control(%{{.*}} : i32, %{{.*}} : i32) = (%{{.*}}, %{{.*}} : i32, i32) to (%{{.*}}, %{{.*}} : i32, i32) step (%{{.*}}, %{{.*}} : i32, i32) {
! CCYCLE:         scf.execute_region no_inline {
! CCYCLE:             cf.cond_br
! CCYCLE-NOT:         acc.loop
! CCYCLE:           acc.yield
! CCYCLE-NEXT:    } inclusiveUpperbound({{.*}}) collapse([2]) collapseDeviceType({{.*}}) independent

!--- cache_select_case.f90

! `acc loop` with `cache` directive and SELECT CASE inside the body - the
! SELECT CASE makes the loop's body have unstructured CFG. Orphan loop
! inside a (non-seq) acc routine defaults to `independent`.
subroutine test_cache_nonunit_lb()
  !$acc routine gang
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
end subroutine

! One loop, bounds on the op, and the SELECT CASE that makes the body branch
! folded into a region along with the cache directive it holds.
! CCASE-LABEL: func.func @_QPtest_cache_nonunit_lb
! CCASE:         acc.loop private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CCASE:         scf.execute_region no_inline {
! CCASE:             acc.cache var({{.*}}) name("arr(15)")
! CCASE:             fir.select_case %{{.*}} : i32 [#fir.point, %{{.*}}, ^{{.*}}, #fir.point, %{{.*}}, ^{{.*}}, unit, ^{{.*}}]
! CCASE-NOT:         acc.loop
! CCASE:           acc.yield
! CCASE-NEXT:    } inclusiveUpperbound({{.*}}) independent
