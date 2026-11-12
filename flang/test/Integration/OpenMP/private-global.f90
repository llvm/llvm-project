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
! The private copy of `table` is a constant-size array of trivial elements, so
! it is privatized unboxed as a plain [10 x i32] on the stack -- no descriptor
! and no heap allocation for the private itself.
! CHECK:         %[[PRIV_TABLE:.*]] = alloca [10 x i32], align 4

! check that we use the private copy of table for table/=50
! CHECK:       omp.par.region7:
! CHECK:         %[[ELT_PTR:.*]] = getelementptr {{.*}} i32, ptr %[[PRIV_TABLE]], i64 %{{.*}}
! CHECK:         %[[ELT:.*]] = load i32, ptr %[[ELT_PTR]]
! CHECK:         icmp ne i32 %[[ELT]], 50

! check that the inline assignment loop stores into the private copy of table
! CHECK:       omp.par.region3:
! CHECK:         %[[ASSIGN_ELT_PTR:.*]] = getelementptr {{.*}} i32, ptr %[[PRIV_TABLE]], i64 %{{.*}}
! CHECK-NEXT:    store i32 50, ptr %[[ASSIGN_ELT_PTR]], align 4
