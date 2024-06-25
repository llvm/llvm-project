! REQUIRES: openmp_runtime

! This test checks lowering of atomic and atomic update constructs
! RUN: bbc %openmp_flags -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

program OmpAtomicUpdate
    use omp_lib
!CHECK: %[[VAL_A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_A_DECLARE:.*]]:2 = hlfir.declare %[[VAL_A]] {{.*}}
!CHECK: %[[VAL_B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_B_DECLARE:.*]]:2 = hlfir.declare %[[VAL_B]] {{.*}}
!CHECK: %[[VAL_C_ADDRESS:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
!CHECK: %[[VAL_C_DECLARE:.*]]:2 = hlfir.declare %[[VAL_C_ADDRESS]] {{.*}}
!CHECK: %[[VAL_D_ADDRESS:.*]] = fir.address_of(@_QFEd) : !fir.ref<i32>
!CHECK: %[[VAL_D_DECLARE:.*]]:2 = hlfir.declare %[[VAL_D_ADDRESS]] {{.}}
!CHECK: %[[VAL_i1_ALLOCA:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFEi1"}
!CHECK: %[[VAL_i1_DECLARE:.*]]:2 = hlfir.declare %[[VAL_i1_ALLOCA]] {{.*}}
!CHECK: %[[VAL_c5:.*]] = arith.constant 5 : index
!CHECK: %[[VAL_K_ALLOCA:.*]] = fir.alloca !fir.array<5xi32> {bindc_name = "k", uniq_name = "_QFEk"}
!CHECK: %[[VAL_K_SHAPED:.*]] = fir.shape %[[VAL_c5]] : (index) -> !fir.shape<1>
!CHECK: %[[VAL_K_DECLARE:.*]]:2 = hlfir.declare %[[VAL_K_ALLOCA]](%[[VAL_K_SHAPED]]) {{.*}}

!CHECK: %[[VAL_W_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFEw"}
!CHECK: %[[VAL_W_DECLARE:.*]]:2 = hlfir.declare %[[VAL_W_ALLOCA]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_X_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAL_X_DECLARE:.*]]:2 = hlfir.declare %[[VAL_X_ALLOCA]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_Y_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[VAL_Y_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Y_ALLOCA]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_Z_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[VAL_Z_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Z_ALLOCA]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    integer :: w, x, y, z
    integer, pointer :: a, b
    integer, target :: c, d
    integer(1) :: i1
    integer, dimension(5) :: k

!CHECK: %[[EMBOX:.*]] = fir.embox %[[VAL_C_DECLARE]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_A_DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[VAL_D_DECLARE]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_B_DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    a=>c
    b=>d

!CHECK: %[[VAL_c3:.*]] = arith.constant 3 : index
!CHECK: %[[VAL_K_DESIGNATE:.*]] = hlfir.designate %[[VAL_K_DECLARE]]#0 (%[[VAL_c3]]) : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[LOADED_Z:.*]] = fir.load %[[VAL_Z_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.update %[[VAL_K_DESIGNATE]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:     %[[TEMP:.*]] = arith.muli %[[LOADED_Z]], %[[ARG]] : i32
!CHECK:     omp.yield(%[[TEMP]] : i32)
!CHECK: }
   !$omp atomic update
        k(3) = z * k(3)

!CHECK: %[[VAL_A_LOADED:.*]] = fir.load %[[VAL_A_DECLARE]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_A_BOX_ADDR:.*]] = fir.box_addr %[[VAL_A_LOADED]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[VAL_B_LOADED:.*]] = fir.load %[[VAL_B_DECLARE]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_B_BOX_ADDR:.*]] = fir.box_addr %[[VAL_B_LOADED]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[VAL_B:.*]] = fir.load %[[VAL_B_BOX_ADDR]] : !fir.ptr<i32>
!CHECK: omp.atomic.update %[[VAL_A_BOX_ADDR]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.addi %[[ARG]], %[[VAL_B]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
   !$omp atomic update
        a = a + b 

!CHECK: %[[VAL_c1:.*]] = arith.constant 1 : i32
!CHECK: omp.atomic.update %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.addi %[[ARG]], %[[VAL_c1]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
   !$omp atomic 
        y = y + 1

!CHECK: %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.update %[[VAL_Z_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.muli %[[VAL_X_LOADED]], %[[ARG]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
    !$omp atomic update
        z = x * z 

!CHECK: %[[VAL_c1:.*]] = arith.constant 1 : i32
!CHECK: omp.atomic.update memory_order(relaxed) hint(uncontended) %[[VAL_X_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.subi %[[ARG]], %[[VAL_c1]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
   !$omp atomic relaxed update hint(omp_sync_hint_uncontended)
        x = x - 1

!CHECK:  %[[VAL_C_LOADED:.*]] = fir.load %[[VAL_C_DECLARE]]#0 : !fir.ref<i32>
!CHECK:  %[[VAL_D_LOADED:.*]] = fir.load %[[VAL_D_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.update memory_order(relaxed) %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK:  {{.*}} = arith.cmpi sgt, %[[ARG]], {{.*}} : i32
!CHECK:  {{.*}} = arith.select {{.*}}, %[[ARG]], {{.*}} : i32
!CHECK:  {{.*}} = arith.cmpi sgt, {{.*}}
!CHECK:  %[[TEMP:.*]] = arith.select {{.*}} : i32
!CHECK:  omp.yield(%[[TEMP]] : i32)
!CHECK: }
    !$omp atomic update relaxed 
        y = max(y, c, d)

!CHECK: %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.update memory_order(relaxed) hint(contended) %[[VAL_Z_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.addi %[[ARG]], %[[VAL_X_LOADED]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
    !$omp atomic relaxed hint(omp_sync_hint_contended)
        z = z + x

!CHECK: %[[VAL_c10:.*]] = arith.constant 10 : i32
!CHECK: omp.atomic.update memory_order(release) hint(contended) %[[VAL_Z_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.muli %[[VAL_c10]], %[[ARG]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
    !$omp atomic release update hint(omp_lock_hint_contended)
        z = z * 10

!CHECK: %[[VAL_Z_LOADED:.*]] = fir.load %[[VAL_Z_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.update memory_order(release) hint(speculative) %[[VAL_X_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.divsi %[[ARG]], %[[VAL_Z_LOADED]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
    !$omp atomic hint(omp_lock_hint_speculative) update release
        x = x / z

!CHECK: %[[VAL_c1:.*]] = arith.constant 1 : i32
!CHECK: omp.atomic.update %[[VAL_i1_DECLARE]]#1 : !fir.ref<i8> {
!CHECK: ^bb0(%[[ARG:.*]]: i8):
!CHECK: %[[CONVERT:.*]] = fir.convert %[[ARG]] : (i8) -> i32
!CHECK: %[[ADD:.*]] = arith.addi %[[CONVERT]], %[[VAL_c1]] : i32
!CHECK: %[[TEMP:.*]] = fir.convert %[[ADD]] : (i32) -> i8
!CHECK: omp.yield(%[[TEMP]] : i8)
!CHECK: }
   !$omp atomic
      i1 = i1 + 1
    !$omp end atomic

!CHECK:   %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK:   omp.atomic.update %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_Y:.*]]: i32):
!CHECK:     %[[Y_UPDATE_VAL:.*]] = arith.andi %[[VAL_X_LOADED]], %[[ARG_Y]] : i32
!CHECK:     omp.yield(%[[Y_UPDATE_VAL]] : i32)
!CHECK:   }
  !$omp atomic update
    y = iand(x,y)

!CHECK:   %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK:   omp.atomic.update %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_Y:.*]]: i32):
!CHECK:     %[[Y_UPDATE_VAL:.*]] = arith.xori %[[VAL_X_LOADED]], %[[ARG_Y]] : i32
!CHECK:     omp.yield(%[[Y_UPDATE_VAL]] : i32)
!CHECK:   }
  !$omp atomic update
    y = ieor(x,y)

!CHECK:   %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK:   %[[VAL_Y_LOADED:.*]] = fir.load %[[VAL_Y_DECLARE]]#0 : !fir.ref<i32>
!CHECK:   %[[VAL_Z_LOADED:.*]] = fir.load %[[VAL_Z_DECLARE]]#0 : !fir.ref<i32>
!CHECK:   omp.atomic.update %[[VAL_W_DECLARE]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_W:.*]]: i32):
!CHECK:     %[[WX_CMP:.*]] = arith.cmpi sgt, %[[ARG_W]], %[[VAL_X_LOADED]] : i32
!CHECK:     %[[WX_MIN:.*]] = arith.select %[[WX_CMP]], %[[ARG_W]], %[[VAL_X_LOADED]] : i32
!CHECK:     %[[WXY_CMP:.*]] = arith.cmpi sgt, %[[WX_MIN]], %[[VAL_Y_LOADED]] : i32
!CHECK:     %[[WXY_MIN:.*]] = arith.select %[[WXY_CMP]], %[[WX_MIN]], %[[VAL_Y_LOADED]] : i32
!CHECK:     %[[WXYZ_CMP:.*]] = arith.cmpi sgt, %[[WXY_MIN]], %[[VAL_Z_LOADED]] : i32
!CHECK:     %[[WXYZ_MIN:.*]] = arith.select %[[WXYZ_CMP]], %[[WXY_MIN]], %[[VAL_Z_LOADED]] : i32
!CHECK:     omp.yield(%[[WXYZ_MIN]] : i32)
!CHECK:   }
  !$omp atomic update
    w = max(w,x,y,z)

end program OmpAtomicUpdate
