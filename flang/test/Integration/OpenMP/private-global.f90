!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

! Regression test for https://github.com/llvm/llvm-project/issues/106297

program bug
  implicit none
  integer :: table(10)
  !$OMP PARALLEL PRIVATE(table)
    table = 50
    if (any(table/=50)) then
      stop 'fail 3'
    end if
  !$OMP END PARALLEL
  print *,'ok'
End Program


! CHECK-LABEL: define internal void {{.*}}..omp_par(
! CHECK:       omp.par.entry:
! CHECK:         %[[PRIV_BOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! ...
! check that the private copy is allocated via malloc
! CHECK:       omp.private.init:
! CHECK:         %[[PRIV_TABLE:.*]] = call ptr @malloc(i64 40)
! ...
! check that we use the private copy of table for the assignment (table = 50)
! The assignment is now inlined as a loop instead of calling _FortranAAssign.
! CHECK:       omp.par.region1:
! CHECK:         call void @llvm.memcpy.p0.p0.i32(ptr{{.*}}%[[BOX_COPY:.*]], ptr{{.*}}%[[PRIV_BOX_ALLOC]], i32 48, i1 false)
! ...
! check that we use the private copy of table for table/=50 (inlined loop body)
! CHECK:       omp.par.region6:
! CHECK:         %[[VAL_44:.*]] = sub {{.*}} i64 %{{.*}}, 1
! ...
! check that we store 50 into the private table's elements (inlined loop body)
! CHECK:       omp.par.region3:
! CHECK:         store i32 50, ptr %{{.*}}, align 4
