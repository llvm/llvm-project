! This test checks lowering of atomic and atomic update constructs
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

program OmpAtomicUpdate
    use omp_lib
    integer :: x, y, z
    integer, pointer :: a, b
    integer, target :: c, d
    a=>c
    b=>d

!CHECK: func.func @_QQmain() attributes {fir.bindc_name = "ompatomicupdate"} {
!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[A_ADDR:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFEa.addr"}
!CHECK: %{{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: fir.store %{{.*}} to %[[A_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[B_ADDR:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFEb.addr"}
!CHECK: %{{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: fir.store %{{.*}} to %[[B_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK: %[[C_ADDR:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
!CHECK: %[[D_ADDR:.*]] = fir.address_of(@_QFEd) : !fir.ref<i32>
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %{{.*}} = fir.convert %[[C_ADDR]] : (!fir.ref<i32>) -> !fir.ptr<i32>
!CHECK: fir.store %{{.*}} to %[[A_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK: %{{.*}} = fir.convert %[[D_ADDR]] : (!fir.ref<i32>) -> !fir.ptr<i32>
!CHECK: fir.store {{.*}} to %[[B_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK: %[[LOADED_A:.*]] = fir.load %[[A_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK:  omp.atomic.update   %[[LOADED_A]] : !fir.ptr<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_B:.*]] = fir.load %[[B_ADDR]] : !fir.ref<!fir.ptr<i32>>
!CHECK:    %{{.*}} = fir.load %[[LOADED_B]] : !fir.ptr<i32>
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], %{{.*}} : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK: }
    !$omp atomic update
        a = a + b 

!CHECK: omp.atomic.update   %[[Y]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    {{.*}} = arith.constant 1 : i32
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], {{.*}} : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   %[[Z]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.muli %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
    !$omp atomic 
        y = y + 1
    !$omp atomic update
        z = x * z 

!CHECK:  omp.atomic.update   memory_order(relaxed) hint(uncontended) %[[X]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 1 : i32
!CHECK:    %[[RESULT:.*]] = arith.subi %[[ARG]], {{.*}} : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(relaxed) %[[Y]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK:    %[[LOADED_Z:.*]] = fir.load %[[Z]] : !fir.ref<i32>
!CHECK:    %{{.*}} = arith.cmpi sgt, %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    %{{.*}} = arith.select %{{.*}}, %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    %{{.*}} = arith.cmpi sgt, %{{.*}}, %[[LOADED_Z]] : i32
!CHECK:    %[[RESULT:.*]] = arith.select %{{.*}}, %{{.*}}, %[[LOADED_Z]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(relaxed) hint(contended) %[[Z]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_X:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], %[[LOADED_X]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
    !$omp atomic relaxed update hint(omp_sync_hint_uncontended)
        x = x - 1
    !$omp atomic update relaxed 
        y = max(x, y, z)
    !$omp atomic relaxed hint(omp_sync_hint_contended)
        z = z + x

!CHECK:  omp.atomic.update   memory_order(release) hint(contended) %[[Z]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 10 : i32
!CHECK:   %[[RESULT:.*]] = arith.muli {{.*}}, %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(release) hint(speculative) %[[X]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_Z:.*]] = fir.load %[[Z]] : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.divsi %[[ARG]], %[[LOADED_Z]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }

    !$omp atomic release update hint(omp_lock_hint_contended)
        z = z * 10
    !$omp atomic hint(omp_lock_hint_speculative) update release
        x = x / z

!CHECK:  omp.atomic.update   memory_order(seq_cst) hint(nonspeculative) %[[Y]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %{{.*}} = arith.constant 10 : i32
!CHECK:    %[[RESULT:.*]] = arith.addi %{{.*}}, %[[ARG]] : i32
!CHECK:   omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  omp.atomic.update   memory_order(seq_cst) %[[Z]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[LOADED_Y:.*]] = fir.load %[[Y]] : !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = arith.addi %[[LOADED_Y]], %[[ARG]] : i32
!CHECK:    omp.yield(%[[RESULT]] : i32)
!CHECK:  }
!CHECK:  return
!CHECK: }
    !$omp atomic hint(omp_sync_hint_nonspeculative) seq_cst
        y = 10 + y
    !$omp atomic seq_cst update
        z = y + z
end program OmpAtomicUpdate
