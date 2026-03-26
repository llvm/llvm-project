! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!
! Checks:
!  - C_PTR mappings expand to `__address` member with CLOSE under USM paths.
!  - use_device_ptr does not implicitly expand member operands in the clause.

subroutine only_cptr_use_device_ptr
  use iso_c_binding
  type(c_ptr) :: cptr
  integer :: i

  !$omp target data use_device_ptr(cptr) map(tofrom: i)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPonly_cptr_use_device_ptr()
! CHECK: %[[I_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "i"}
! CHECK: %[[CP_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<{{.*}}__builtin_c_ptr{{.*}}>>, !fir.type<{{.*}}__builtin_c_ptr{{.*}}>) map_clauses(return_param) capture(ByRef) -> !fir.ref<!fir.type<{{.*}}__builtin_c_ptr{{.*}}>>
! CHECK: omp.target_data map_entries(%[[I_MAP]] : !fir.ref<i32>) use_device_ptr(%[[CP_MAP]] -> %{{.*}} : !fir.ref<!fir.type<{{.*}}__builtin_c_ptr{{.*}}>>) {
! CHECK:   omp.terminator
! CHECK: }
