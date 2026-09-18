!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s

! End-to-end coverage: Flang lowers a taskgroup task_reduction enclosing a
! target in_reduction all the way to LLVM IR. Verifies that the host path
! emits __kmpc_taskred_init for the enclosing taskgroup and
! __kmpc_task_reduction_get_th_data for the target's in_reduction lookup,
! with load/store through the returned private pointer and no direct update
! through the original shared pointer.

subroutine target_in_reduction_e2e()
  integer :: i
  i = 0
  !$omp taskgroup task_reduction(+:i)
    !$omp target in_reduction(+:i)
    i = i + 1
    !$omp end target
  !$omp end taskgroup
end subroutine target_in_reduction_e2e

! CHECK-LABEL: define void @target_in_reduction_e2e_()
! The enclosing taskgroup emits __kmpc_taskred_init to register the
! task_reduction descriptor.
! CHECK:         call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %{{.+}})

! The host stub calls the outlined target body passing the captured pointer.
! CHECK:         call void @__omp_offloading_{{.*}}_target_in_reduction_e2e_{{.*}}(ptr %{{.+}}, ptr null)

! Inside the outlined target body, the in_reduction private pointer is
! obtained from the runtime using the captured original pointer with a NULL
! descriptor (the runtime walks enclosing taskgroups). All loads and stores
! go through the returned private pointer.
! CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_in_reduction_e2e_
! CHECK-SAME:    (ptr %[[ORIG:.+]], ptr %{{.+}})
! CHECK:         %[[GTID:.+]] = call i32 @__kmpc_global_thread_num(
! CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[ORIG]])
! CHECK:         %[[VAL:.+]] = load i32, ptr %[[PRIV]]
! CHECK:         %[[SUM:.+]] = add i32 %[[VAL]], 1
! CHECK:         store i32 %[[SUM]], ptr %[[PRIV]]

! The outlined body must not store directly through the captured original
! pointer; all updates go through the runtime-returned private copy.
! CHECK-NOT:     store i32 %{{.+}}, ptr %[[ORIG]]
