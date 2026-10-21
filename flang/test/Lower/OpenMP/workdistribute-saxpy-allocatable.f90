! RUN: %flang_fc1 -emit-fir -O0 -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O3 -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Allocatable descriptors crossing the workdistribute target fission are
! reloaded from their maps in the kernel, not cached by value.

! CHECK-LABEL: func @_QPsaxpy_allocatable
! CHECK:         %[[Y_MAP:.*]] = omp.map.info {{.*}}map_clauses(storage) capture(ByRef) members({{.*}}) name("y")

! Scalars and temp array pointers may be cached, but no descriptor is.
! CHECK:         omp.target_data
! CHECK-NOT:     !fir.box<{{.*}}name("__flang_workdistribute

! The kernel computing a * x + y reloads y's box from its mapped descriptor.
! CHECK:         omp.target kernel_type(spmd) {{.*}}map_entries(%[[Y_MAP]] -> %[[Y_ARG:[^,]+]],
! CHECK:           %[[Y_DECL:.*]] = fir.declare %[[Y_ARG]] uniq_name("_QFsaxpy_allocatableEy")
! CHECK:           %[[Y_BOX:.*]] = fir.load %[[Y_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK-NEXT:          omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest
! CHECK:                     fir.box_addr %[[Y_BOX]]

! Kernels after the host realloc store still reload y from its map.
! CHECK-NOT:     !fir.box<{{.*}}name("__flang_workdistribute
! CHECK:         fir.store %{{.*}} to %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         omp.target kernel_type(spmd) {{.*}}map_entries(%[[Y_MAP]] -> %[[Y_COPY_ARG:[^,]+]],
! CHECK:           %[[Y_COPY_DECL:.*]] = fir.declare %[[Y_COPY_ARG]] uniq_name("_QFsaxpy_allocatableEy")
! CHECK:           fir.load %[[Y_COPY_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK-NEXT:          omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest
subroutine saxpy_allocatable(a, x, y)
  real :: a
  real, allocatable :: x(:), y(:)

  !$omp target teams workdistribute
  y = a * x + y
  !$omp end target teams workdistribute
end subroutine saxpy_allocatable
