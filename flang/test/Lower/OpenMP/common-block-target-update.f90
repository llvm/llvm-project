!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program main
    implicit none
    common/cb_array/array
    real(kind=8), dimension(100) :: array
    integer i

    !$omp target update to(/cb_array/)

    !$omp target teams distribute parallel do map(tofrom: /cb_array/)
        do i = 1, 100
            array(i) = i
        enddo
    !$omp end target teams distribute parallel do

    !$omp target update from(/cb_array/)
end program

! CHECK-LABEL: func.func @_QQmain()

! CHECK: %[[CB_ADDR:.*]] = fir.address_of(@cb_array_) : !fir.ref<!fir.array<800xi8>>
! CHECK: %[[MAP_TO:.*]] = omp.map.info var_ptr(%[[CB_ADDR]] : !fir.ref<!fir.array<800xi8>>, !fir.array<800xi8>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.array<800xi8>> {name = "cb_array"}
! CHECK: omp.target_update map_entries(%[[MAP_TO]] : !fir.ref<!fir.array<800xi8>>)

! CHECK: %[[MAP_TOFROM:.*]] = omp.map.info var_ptr(%[[CB_ADDR]] : !fir.ref<!fir.array<800xi8>>, !fir.array<800xi8>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.array<800xi8>> {name = "cb_array"}
! CHECK: omp.target {{.*}} map_entries(%[[MAP_TOFROM]] -> %{{.*}}, %{{.*}} -> %{{.*}} : !fir.ref<!fir.array<800xi8>>, !fir.ref<i32>)

! CHECK: %[[MAP_FROM:.*]] = omp.map.info var_ptr(%[[CB_ADDR]] : !fir.ref<!fir.array<800xi8>>, !fir.array<800xi8>) map_clauses(from) capture(ByRef) -> !fir.ref<!fir.array<800xi8>> {name = "cb_array"}
! CHECK: omp.target_update map_entries(%[[MAP_FROM]] : !fir.ref<!fir.array<800xi8>>)
