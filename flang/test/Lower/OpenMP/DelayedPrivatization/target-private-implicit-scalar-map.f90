! Tests delayed privatization works for implicit capture of scalars similarly to
! the way it works for explicitly firstprivitized scalars.

! RUN: %flang_fc1 -emit-mlir -fopenmp -mmlir --enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:   omp.private {type = firstprivate} @_QFExdgfx_firstprivate_i32 : i32 copy {
! CHECK:         ^bb0(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>):
! CHECK:           %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK:           omp.yield(%{{.*}} : !fir.ref<i32>)
! CHECK:         }

! CHECK-LABEL:   omp.private {type = firstprivate} @_QFExfpvx_firstprivate_i32 : i32 copy {
! CHECK:         ^bb0(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>):
! CHECK:           %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK:           omp.yield(%{{.*}} : !fir.ref<i32>)
! CHECK:         }

! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "xdgfx", uniq_name = "_QFExdgfx"}
! CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFExdgfx"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:  %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "xfpvx", uniq_name = "_QFExfpvx"}
! CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFExfpvx"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:  %[[VAL_4:.*]] = omp.map.info var_ptr(%[[VAL_3]] : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>

! CHECK:  omp.target map_entries(%[[VAL_4]] -> %{{.*}}, %[[VAL_5]] -> %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) private(@_QFExfpvx_firstprivate_i32 %[[VAL_3]] -> %[[VAL_6:.*]] [map_idx=0], @_QFExdgfx_firstprivate_i32 %[[VAL_1]] -> %[[VAL_7:.*]] [map_idx=1] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:  %{{.*}} = fir.declare %[[VAL_6]] {uniq_name = "_QFExfpvx"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:  %{{.*}} = fir.declare %[[VAL_7]] {uniq_name = "_QFExdgfx"} : (!fir.ref<i32>) -> !fir.ref<i32>

program test_default_implicit_firstprivate
  implicit none
  integer :: xdgfx, xfpvx
  xdgfx = 1
  xfpvx = 2
  !$omp target firstprivate(xfpvx)
  xdgfx = 42
  xfpvx = 43
  !$omp end target
  write(*,*) xdgfx, xfpvx
end program
