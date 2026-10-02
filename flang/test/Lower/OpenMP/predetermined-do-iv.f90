! Test the privatization of predetermined DO loop iteration variables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPparallel_do()
! CHECK:         %[[I3_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_doEi3"}
! CHECK:         omp.parallel private({{.*}}Ei3_private_i32 %[[I3_HOST]]#0 -> %[[I3_PRIV:[^ ]+]] : !fir.ref<i32>) {
! CHECK-NOT:       fir.alloca {{.*}}bindc_name = "i3"
! CHECK:           %[[I3_PRIV_DECL:.*]]:2 = hlfir.declare %[[I3_PRIV]] {uniq_name = "_QFparallel_doEi3"}
! CHECK:           omp.wsloop private({{.*}}Ei2_private_i32 %{{[^#]+}}#0 -> %[[I2_PRIV:[^ ]+]] : !fir.ref<i32>) {
! CHECK:             omp.loop_nest
! CHECK:               hlfir.declare %[[I2_PRIV]] {uniq_name = "_QFparallel_doEi2"}
! CHECK:               fir.do_loop
! CHECK:                 fir.store %{{.*}} to %[[I3_PRIV_DECL]]#0
! CHECK:               omp.yield
! CHECK:           %[[LOAD:.*]] = fir.load %[[I3_PRIV_DECL]]#0
! CHECK:           %[[C3:.*]] = arith.constant 3 : i32
! CHECK:           arith.cmpi ne, %[[LOAD]], %[[C3]] : i32
subroutine parallel_do()
  integer :: i2 = 10, i3 = 99
  integer :: a = 1, b = 1
  !$omp parallel
    !$omp do
      do i2 = 1, 2
        do i3 = 1, 2
          b = b + a
        end do
      end do
    !$omp end do
    if (i3 /= 3) print *, 'Error i3', i3
  !$omp end parallel
end subroutine

! CHECK-LABEL: func @_QPcollapsed_do()
! CHECK:         %[[K_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFcollapsed_doEk"}
! CHECK:         %[[L_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFcollapsed_doEl"}
! CHECK:         omp.parallel private(
! CHECK:             @_QFcollapsed_doEk_private_i32 %[[K_HOST]]#0 -> %[[K_PAR:[a-z0-9]+]]
! CHECK:             @_QFcollapsed_doEl_private_i32 %[[L_HOST]]#0 -> %[[L_PAR:[a-z0-9]+]]
! CHECK-NOT:       fir.alloca {{.*}}bindc_name = "k"
! CHECK-NOT:       fir.alloca {{.*}}bindc_name = "l"
! CHECK:           %[[K_PAR_DECL:.*]]:2 = hlfir.declare %[[K_PAR]] {uniq_name = "_QFcollapsed_doEk"}
! CHECK:           %[[L_PAR_DECL:.*]]:2 = hlfir.declare %[[L_PAR]] {uniq_name = "_QFcollapsed_doEl"}
! CHECK:           omp.wsloop private(@_QFcollapsed_doEi_private_i32 %{{[^,]+}}, @_QFcollapsed_doEj_private_i32 %{{[^:]+}}
! CHECK-SAME:                           : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:             fir.do_loop
! CHECK:               fir.store %{{.*}} to %[[K_PAR_DECL]]#0
! CHECK:               fir.do_loop
! CHECK:                 fir.store %{{.*}} to %[[L_PAR_DECL]]#0
! CHECK:          %[[C33:.*]] = arith.constant 33 : i32
! CHECK:          hlfir.assign %[[C33]] to %[[K_PAR_DECL]]#0
! CHECK:          %[[C44:.*]] = arith.constant 44 : i32
! CHECK:          hlfir.assign %[[C44]] to %[[L_PAR_DECL]]#0
subroutine collapsed_do()
  integer :: i, j, k, l
  !$omp parallel
  !$omp do collapse(2)
  do i = 1, 4
    do j = 1, 4
      do k = 1, 4
        do l = 1, 4
        end do
      end do
    end do
  end do
  !$omp end do
  k = 33
  l = 44
  !$omp end parallel
end subroutine

! CHECK-LABEL: func @_QPnested_parallel()
! CHECK:         %[[J_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnested_parallelEj"}
! CHECK:         omp.parallel {
! CHECK:           omp.parallel private(@_QFnested_parallelEj_private_i32 %[[J_HOST]]#0 -> %[[J_PAR:[a-z0-9]+]] : !fir.ref<i32>) {
! CHECK:             %[[J_PAR_DECL:.*]]:2 = hlfir.declare %[[J_PAR]] {uniq_name = "_QFnested_parallelEj"}
! CHECK:             fir.do_loop
! CHECK:               fir.store %{{.*}} to %[[J_PAR_DECL]]#0
subroutine nested_parallel()
  integer :: j
  !$omp parallel
  !$omp parallel
  do j = 1, 4
  end do
  !$omp end parallel
  !$omp end parallel
end subroutine

! CHECK-LABEL: func @_QPparallel_task()
! CHECK:         %[[K_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_taskEk"}
! CHECK:         omp.parallel {
! CHECK:           omp.single
! CHECK:             omp.task private(@_QFparallel_taskEk_private_i32 %[[K_HOST]]#0 -> %[[K_TASK:[a-z0-9]+]] : !fir.ref<i32>) {
! CHECK:               %[[K_TASK_DECL:.*]]:2 = hlfir.declare %[[K_TASK]] {uniq_name = "_QFparallel_taskEk"}
! CHECK:               fir.do_loop
! CHECK:                 fir.store %{{.*}} to %[[K_TASK_DECL]]#0
! CHECK:               %[[C11:.*]] = arith.constant 11 : i32
! CHECK:               hlfir.assign %[[C11]] to %[[K_TASK_DECL]]#0
! CHECK:             %[[C22:.*]] = arith.constant 22 : i32
! CHECK:             hlfir.assign %[[C22]] to %[[K_HOST]]#0
subroutine parallel_task()
  integer :: k
  !$omp parallel
  !$omp single
  !$omp task
  do k = 1, 4
  end do
  k = 11
  !$omp end task
  k = 22
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func @_QPparallel_single()
! CHECK:         %[[J_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_singleEj"}
! CHECK:         omp.parallel private(@_QFparallel_singleEj_private_i32 %[[J_HOST]]#0 -> %[[J_PAR:[a-z0-9]+]] : !fir.ref<i32>) {
! CHECK:           %[[J_PAR_DECL:.*]]:2 = hlfir.declare %[[J_PAR]] {uniq_name = "_QFparallel_singleEj"}
! CHECK:           omp.single {
! CHECK-NOT:         fir.alloca {{.*}}bindc_name = "j"
! CHECK-NOT:         hlfir.declare {{.*}}uniq_name = "_QFparallel_singleEj"
! CHECK:             fir.do_loop
! CHECK:               fir.store %{{.*}} to %[[J_PAR_DECL]]#0
subroutine parallel_single()
  integer :: j
  !$omp parallel
  !$omp single
  do j = 1, 4
  end do
  !$omp end single
  !$omp end parallel
end subroutine

! XXX STOPPED HERE

! CHECK-LABEL: func @_QPblock_local_iv()
! CHECK:         omp.parallel {
! CHECK:           %[[I_ALLOCA:.*]] = fir.alloca i32 {{.*}}bindc_name = "i"
! CHECK:           %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ALLOCA]] {uniq_name = "{{.*}}Ei"}
! CHECK:           fir.do_loop
! CHECK:             fir.store %{{.*}} to %[[I_DECL]]#0
subroutine block_local_iv()
  !$omp parallel
  block
    integer :: i
    do i = 1, 4
    end do
  end block
  !$omp end parallel
end subroutine

! CHECK-LABEL: func @_QPorphaned_do(
! CHECK:         %[[I_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QForphaned_doEi"}
! CHECK:         %[[J_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QForphaned_doEj"}
! CHECK:         omp.wsloop private(@_QForphaned_doEi_private_i32 %[[I_HOST]]#0 ->
! CHECK-SAME:          %[[I_PRIV:[a-z0-9]+]] : !fir.ref<i32>) {
! CHECK:         omp.loop_nest (%[[ARG:[a-z0-9]+]])
! CHECK-NOT:       hlfir.declare {{.*}}uniq_name = "_QForphaned_doEj"
! CHECK:           %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV]] {uniq_name = "_QForphaned_doEi"
! CHECK:           hlfir.assign %[[ARG]] to %[[I_PRIV_DECL]]#0
! CHECK:           fir.do_loop
! CHECK:             fir.store %{{.*}} to %[[J_HOST]]#0
subroutine orphaned_do()
  integer :: i, j
  !$omp do
  do i = 1, 4
    do j = 1, 4
    end do
  end do
  !$omp end do
end subroutine

! CHECK-LABEL: func @_QPparallel_dos()
! CHECK:         %[[K_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_dosEk"}
! CHECK:         omp.parallel private(@_QFparallel_dosEi_private_i32 {{[^,]+}},
! CHECK-SAME:        @_QFparallel_dosEk_private_i32 %[[K_HOST]]#0 -> %[[K_PAR:[^ ]+]]
! CHECK-SAME:        : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:           %[[K_PAR_DECL:.*]]:2 = hlfir.declare %[[K_PAR]] {uniq_name = "_QFparallel_dosEk"}
! CHECK:           fir.do_loop
! CHECK:             omp.wsloop private(@_QFparallel_dosEk_private_i32 %[[K_PAR_DECL]]#0 ->
! CHECK-SAME:            %[[K_WSLOOP:[^ ]+]] : !fir.ref<i32>) {
! CHECK:               omp.loop_nest (%[[ARG1:[^)]*]])
! CHECK:                 %[[K_WSLOOP_DECL:.*]]:2 = hlfir.declare %[[K_WSLOOP]] {uniq_name = "_QFparallel_dosEk"}
! CHECK:                 hlfir.assign %[[ARG1]] to %[[K_WSLOOP_DECL]]#0
! CHECK:             omp.wsloop private(@_QFparallel_dosEj_private_i32 {{[^,:]*}} : !fir.ref<i32>) {
! CHECK-NOT:           fir.alloca {{.*}}bindc_name = "k"
! CHECK-NOT:           hlfir.declare {{.*}}uniq_name = "_QFparallel_dosEk"
! CHECK:               omp.loop_nest
! CHECK:                 fir.do_loop
! CHECK:                   fir.store %{{.*}} to %[[K_PAR_DECL]]#0
! CHECK:           omp.terminator
subroutine parallel_dos()
  integer :: i, j, k
  !$omp parallel
    do i = 1, 2
      !$omp do
        do k = 1, 3
        enddo
      !$omp end do
      !$omp do
        do j = 1, 2
          do k = 1, 3
          enddo
        enddo
      !$omp end do
    enddo
  !$omp end parallel
end subroutine
