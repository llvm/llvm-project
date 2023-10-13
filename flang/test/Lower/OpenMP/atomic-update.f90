! This test checks lowering of atomic and atomic update constructs
! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program OmpAtomicUpdate
    use omp_lib
    integer :: x, y, z
    integer, pointer :: a, b
    integer, target :: c, d
    integer(1) :: i1

    a=>c
    b=>d
    
!CHECK: func.func @_QQmain() attributes {fir.bindc_name = "ompatomicupdate"} {
!CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[A_ADDR:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[VAL_5:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[B_ADDR:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[VAL_8:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
!CHECK: %[[C_ADDR:.*]]:2 = hlfir.declare %[[VAL_8]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_10:.*]] = fir.address_of(@_QFEd) : !fir.ref<i32>
!CHECK: %[[D_ADDR:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEd"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_12:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFEi1"}
!CHECK: %[[I1:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK: %[[VAL_174:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[VAL_174]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_176:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y:.*]]:2 = hlfir.declare %[[VAL_176]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_178:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[Z:.*]]:2 = hlfir.declare %[[VAL_178]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_180:.*]] = fir.embox %[[C_ADDR]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[VAL_180]] to %[[A_ADDR]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_181:.*]] = fir.embox %[[D_ADDR]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[VAL_181]] to %[[B_ADDR]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_182:.*]] = fir.load %[[A_ADDR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_183:.*]] = fir.box_addr %[[VAL_182]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[LOADED_A:.*]]:2 = hlfir.declare %[[VAL_183]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEa"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
!CHECK: omp.atomic.update   %[[LOADED_A]]#0 : !fir.ptr<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:  %[[LOADED_B:.*]] = fir.load %[[B_ADDR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:  %[[VAL_186:.*]] = fir.box_addr %[[LOADED_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:  %{{.*}} = fir.load %[[VAL_186]] : !fir.ptr<i32>
!CHECK:  %[[RESULT:.*]] = arith.addi %[[ARG]], %{{.*}} : i32
!CHECK:  omp.yield(%[[RESULT]] : i32)
!CHECK: }
    !$omp atomic update
        a = a + b 

!CHECK: omp.atomic.update   %[[Y]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    {{.*}} = arith.constant 1 : i32
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], {{.*}} : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   %[[Z]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.muli %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
    !$omp atomic 
        y = y + 1
    !$omp atomic update
        z = x * z 

!CHECK:  omp.atomic.update   memory_order(relaxed) hint(uncontended) %[[X]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 1 : i32
!CHECK:    %[[RESULT:.*]] = arith.subi %[[ARG]], {{.*}} : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(relaxed) %[[Y]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
!CHECK:    %[[LOADED_Z:.*]] = fir.load %[[Z]]#0 : !fir.ref<i32>
!CHECK:    %{{.*}} = arith.cmpi sgt, %[[ARG]], %[[LOADED_X]] : i32
!CHECK:    %{{.*}} = arith.select %{{.*}}, %[[ARG]], %[[LOADED_X]] : i32
!CHECK:    %{{.*}} = arith.cmpi sgt, %{{.*}}, %[[LOADED_Z]] : i32
!CHECK:    %[[RESULT:.*]] = arith.select %{{.*}}, %{{.*}}, %[[LOADED_Z]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(relaxed) hint(contended) %[[Z]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], %[[LOADED_X]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
    !$omp atomic relaxed update hint(omp_sync_hint_uncontended)
        x = x - 1
    !$omp atomic update relaxed 
        y = max(y, x, z)
    !$omp atomic relaxed hint(omp_sync_hint_contended)
        z = z + x

!CHECK:  omp.atomic.update   memory_order(release) hint(contended) %[[Z]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 10 : i32
!CHECK:   %[[RESULT:.*]] = arith.muli {{.*}}, %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(release) hint(speculative) %[[X]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_Z:.*]] = fir.load %[[Z]]#0 : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.divsi %[[ARG]], %[[LOADED_Z]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }

    !$omp atomic release update hint(omp_lock_hint_contended)
        z = z * 10
    !$omp atomic hint(omp_lock_hint_speculative) update release
        x = x / z

!CHECK:  omp.atomic.update   memory_order(seq_cst) hint(nonspeculative) %[[Y]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 10 : i32
!CHECK:    %[[RESULT:.*]] = arith.addi %{{.*}}, %[[ARG]] : i32
!CHECK:   omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(seq_cst) %[[Z]]#0 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_Y:.*]] = fir.load %[[Y]]#0 : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.addi %[[LOADED_Y]], %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
    !$omp atomic hint(omp_sync_hint_nonspeculative) seq_cst
        y = 10 + y
    !$omp atomic seq_cst update
        z = y + z

!CHECK:  omp.atomic.update   %[[I1]]#0 : !fir.ref<i8> {
!CHECK:  ^bb0(%[[VAL:.*]]: i8):
!CHECK:    %[[CVT_VAL:.*]] = fir.convert %[[VAL]] : (i8) -> i32
!CHECK:    %[[C1_VAL:.*]] = arith.constant 1 : i32
!CHECK:    %[[ADD_VAL:.*]] = arith.addi %[[CVT_VAL]], %[[C1_VAL]] : i32
!CHECK:    %[[UPDATED_VAL:.*]] = fir.convert %[[ADD_VAL]] : (i32) -> i8
!CHECK:    omp.yield(%[[UPDATED_VAL]] : i8)
!CHECK:  }
    !$omp atomic
      i1 = i1 + 1
    !$omp end atomic
!CHECK:  return
!CHECK: }
end program OmpAtomicUpdate
