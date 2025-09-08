! Test that we appropriately categorize types as firstprivate even across
! module boundaries.

!RUN: split-file %s %t

!RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging -fopenmp-version=50 %t/imp_scalar_map_module.f90 -o - \
!RUN: | %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging  -fopenmp-version=50 %t/imp_scalar_map_target.f90 -o - \
!RUN: | FileCheck %t/imp_scalar_map_target.f90

!--- imp_scalar_map_module.f90
module test_data
    implicit none
    integer :: z
    real :: i(10,10), j(5,5,2), k(25,2)
    equivalence(j(1,1,1),k(1,1))
end module

!--- imp_scalar_map_target.f90
subroutine target_imp_capture
    use test_data
    implicit none
    integer :: x, y

    !$omp target map(tofrom: x)
        x = y + z + i(1,1) + j(1,1,1) + k(1,1)
    !$omp end target

end subroutine target_imp_capture

! CHECK-LABEL: func.func @_QPtarget_imp_capture()
! CHECK:           %[[VAL_0:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "x"}
! CHECK:           %[[VAL_1:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.array<10x10xf32>>, !fir.array<10x10xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "i"}
! CHECK:           %[[VAL_2:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ptr<!fir.array<5x5x2xf32>>, !fir.array<5x5x2xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ptr<!fir.array<5x5x2xf32>> {name = "j"}
! CHECK:           %[[VAL_3:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ptr<!fir.array<25x2xf32>>, !fir.array<25x2xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ptr<!fir.array<25x2xf32>> {name = "k"}
! CHECK:           %[[VAL_4:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
! CHECK:           omp.target map_entries(%[[VAL_0]] -> %[[VAL_6:.*]], %[[VAL_1]] -> %[[VAL_7:.*]], %[[VAL_2]] -> %[[VAL_8:.*]], %[[VAL_3]] -> %[[VAL_9:.*]], %[[VAL_4]] -> %[[VAL_10:.*]], %[[VAL_5]] -> %[[VAL_11:.*]] : !fir.ref<i32>, !fir.ref<!fir.array<10x10xf32>>, !fir.ptr<!fir.array<5x5x2xf32>>, !fir.ptr<!fir.array<25x2xf32>>, !fir.ref<i32>, !fir.ref<i32>) private(@_QFtarget_imp_captureEy_firstprivate_i32 %{{.*}}#0 -> %[[VAL_12:.*]] [map_idx=4], @_QMtest_dataEz_firstprivate_i32 %{{.*}}#0 -> %[[VAL_13:.*]] [map_idx=5] : !fir.ref<i32>, !fir.ref<i32>) {
