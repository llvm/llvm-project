! Tests delayed privatization works for implicit capture of scalars similarly to
! the way it works for explicitly firstprivitized scalars.

! RUN: %flang_fc1 -emit-mlir -fopenmp -mmlir --enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s

!CHECK:   omp.private {type = private} @[[SYM_K:.*]] : i32
!CHECK:   omp.private {type = private} @[[SYM_J:.*]] : i32
!CHECK:   omp.private {type = private} @[[SYM_I:.*]] : i32
!CHECK:   omp.private {type = firstprivate} @[[SYM_XDGFX:.*]] : i32 copy {
!CHECK:   omp.private {type = firstprivate} @[[SYM_XFPVX:.*]] : i32 copy {

program test_default_implicit_firstprivate
  implicit none
  integer :: xdgfx, xfpvx
  integer :: i,j,k
  integer :: arr(10,10,10)
  integer, allocatable :: allocarr(:,:,:)
!CHECK:           %[[VAL_0:.*]] = fir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEallocarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
!CHECK:           %[[VAL_1:.*]] = fir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10x10x10xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<10x10x10xi32>>
!CHECK:           %[[VAL_2:.*]] = fir.declare %{{.*}} {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:           %[[VAL_3:.*]] = fir.declare %{{.*}} {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:           %[[VAL_4:.*]] = fir.declare %{{.*}} {uniq_name = "_QFEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:           %[[VAL_5:.*]] = fir.declare %{{.*}} {uniq_name = "_QFExdgfx"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:           %[[VAL_6:.*]] = fir.declare %{{.*}} {uniq_name = "_QFExfpvx"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:           %[[VAL_7:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
!CHECK:           %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_3]] : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "j"}
!CHECK:           %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "k"}
!CHECK:           %[[VAL_10:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xi32>>>
!CHECK:           %[[VAL_11:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_10]] : !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xi32>>>) bounds({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xi32>>> {name = ""}
!CHECK:           %[[VAL_12:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>, !fir.box<!fir.heap<!fir.array<?x?x?xi32>>>) map_clauses(implicit, to) capture(ByRef) members(%[[VAL_11]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>> {name = "allocarr"}
!CHECK:           %[[VAL_13:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !fir.ref<!fir.array<10x10x10xi32>>, !fir.array<10x10x10xi32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<10x10x10xi32>> {name = "arr"}
!CHECK:           %[[VAL_14:.*]] = omp.map.info var_ptr(%[[VAL_6]] : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
!CHECK:           %[[VAL_15:.*]] = omp.map.info var_ptr(%[[VAL_5]] : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
!CHECK:           omp.target host_eval({{.*}}) map_entries(%[[VAL_7]] -> %{{.*}}, %[[VAL_8]] -> %{{.*}}, %[[VAL_9]] -> %{{.*}}, %[[VAL_12]] -> %{{.*}}, %[[VAL_13]] -> %{{.*}}, %[[VAL_14]] -> %{{.*}}, %[[VAL_15]] -> %{{.*}}, %[[VAL_11]] -> %{{.*}} : {{.*}}) private(@[[SYM_XFPVX]] %[[VAL_6]] -> %{{.*}} [map_idx=5], @[[SYM_XDGFX]] %[[VAL_5]] -> %{{.*}} [map_idx=6] : {{.*}}) {
!CHECK              omp.parallel private(@[[SYM_XFPVX]] %{{.*}} -> %{{.*}}, @[[SYM_XDGFX]] %{{.*}} -> %{{.*}}, @[[SYM_I]] %{{.*}} -> %{{.*}}, @[[SYM_J]] %{{.*}} -> %{{.*}}, @[[SYM_K]] %{{.*}} -> %{{.*}} : {{.*}}) {
  !$omp target teams distribute parallel do collapse(3) firstprivate(xfpvx)
    do i = 1, 10
        do j = 1, 10
            do k = 1, 10
                allocarr(i,j,k) = arr(i,j,k) + xdgfx + xfpvx
            end do
        end do
    end do
end program
