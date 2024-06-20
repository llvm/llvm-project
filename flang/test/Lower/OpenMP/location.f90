! This test checks location of OpenMP constructs and clauses

!RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s

!CHECK-LABEL: sub_parallel
subroutine sub_parallel()
  print *, x
!CHECK: omp.parallel   {
  !$omp parallel
    print *, x
!CHECK:   omp.terminator loc(#[[PAR_LOC:.*]])
!CHECK: } loc(#[[PAR_LOC]])
  !$omp end parallel
  print *, x
end

!CHECK-LABEL: sub_target
subroutine sub_target()
  print *, x
!CHECK: omp.target {{.*}} {
  !$omp target
    print *, x
!CHECK:   omp.terminator loc(#[[TAR_LOC:.*]])
!CHECK: } loc(#[[TAR_LOC]])
  !$omp end target
  print *, x
end

!CHECK-LABEL: sub_loop
subroutine sub_loop()
!CHECK: omp.wsloop {
!CHECK-NEXT: omp.loop_nest {{.*}} {
  !$omp do
  do i=1,10
    print *, i
!CHECK:   omp.yield loc(#[[LOOP_LOC:.*]])
!CHECK: } loc(#[[LOOP_LOC]])
!CHECK:   omp.terminator loc(#[[LOOP_LOC]])
!CHECK: } loc(#[[LOOP_LOC]])
  end do
  !$omp end do
end

!CHECK-LABEL: sub_standalone
subroutine sub_standalone()
  !CHECK: omp.barrier loc(#[[BAR_LOC:.*]])
  !$omp barrier
  !CHECK: omp.taskwait loc(#[[TW_LOC:.*]])
  !$omp taskwait
  !CHECK: omp.taskyield loc(#[[TY_LOC:.*]])
  !$omp taskyield
end

subroutine sub_if(c)
  logical(kind=4) :: c
  !CHECK: %[[CVT:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1 loc(#[[IF_LOC:.*]])
  !CHECK: omp.task if(%[[CVT]])
  !$omp task if(c)
    print *, "Task"
  !$omp end task
  !CHECK: } loc(#[[TASK_LOC:.*]])
end subroutine

!CHECK: #[[PAR_LOC]] = loc("{{.*}}location.f90":9:9)
!CHECK: #[[TAR_LOC]] = loc("{{.*}}location.f90":21:9)
!CHECK: #[[LOOP_LOC]] = loc("{{.*}}location.f90":33:9)
!CHECK: #[[BAR_LOC]] = loc("{{.*}}location.f90":47:9)
!CHECK: #[[TW_LOC]] = loc("{{.*}}location.f90":49:9)
!CHECK: #[[TY_LOC]] = loc("{{.*}}location.f90":51:9)
!CHECK: #[[IF_LOC]] = loc("{{.*}}location.f90":58:14)
!CHECK: #[[TASK_LOC]] = loc("{{.*}}location.f90":58:9)
