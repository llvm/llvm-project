! The "use_device_addr" was added to the "target data" directive in OpenMP 5.0.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! This test primary goal is to check that we update only base addr for
! arrays used in used_device_addr clause.

!CHECK: func.func @{{.*}}device_addr_default(
!CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(always, to, literal) capture(ByRef) name("x") -> !fir.ref<!fir.array<?xi32>>
!CHECK: omp.target_data use_device_addr(%[[MAP]] -> %[[ARG:.*]] : !fir.ref<!fir.array<?xi32>>) {
!CHECK:   %[[ALLOCA_TGT_DESC:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:   %[[ALLOCA_HOST_DESC:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:   fir.store %[[ARG]] to %[[ALLOCA_HOST_DESC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:   %[[LOADED_HOST_DESC:.*]] = fir.load %[[ALLOCA_HOST_DESC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:   %[[HOST_ARR_ADDR:.*]] = fir.box_addr %[[LOADED_HOST_DESC]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
!CHECK:   %[[HOST_ARR_PTR:.*]] = fir.convert %[[HOST_ARR_ADDR]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.llvm_ptr<i8>
!CHECK:   %[[DEVICE_ID:.*]] = fir.call @omp_get_default_device() : () -> i32
!CHECK:   %[[DEVICE_ID_CONV:.*]] = fir.convert %[[DEVICE_ID]] : (i32) -> i64
!CHECK:   %[[PTR_ARG:.*]] = fir.convert %[[HOST_ARR_PTR]] : (!fir.llvm_ptr<i8>) -> !fir.llvm_ptr<i8>
!CHECK:   %[[TGT_PTR:.*]] = fir.call @__tgt_get_mapped_ptr(%[[DEVICE_ID_CONV]], %[[PTR_ARG]]) : (i64, !fir.llvm_ptr<i8>) -> !fir.llvm_ptr<i8>
!CHECK:   %[[TGT_PTR_CONV:.*]] = fir.convert %[[TGT_PTR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<?xi32>>
!CHECK:   %[[C0:.*]] = arith.constant 0 : index
!CHECK:   %[[ARR_DIMS:.*]]:3 = fir.box_dims %[[LOADED_HOST_DESC]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!CHECK:   %[[TGT_DESC:.*]] = fir.create_box %[[TGT_PTR_CONV]] lbs(%[[ARR_DIMS]]#0) extents(%[[ARR_DIMS]]#1) strides(%[[ARR_DIMS]]#2) : (!fir.ref<!fir.array<?xi32>>, index, index, index) -> !fir.box<!fir.array<?xi32>>
!CHECK:   fir.store %[[TGT_DESC]] to %[[ALLOCA_TGT_DESC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:   %[[RES_TGT_DESC:.*]] = fir.load %[[ALLOCA_TGT_DESC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:   %[[DECL:.*]] = hlfir.declare %[[RES_TGT_DESC]] {fortran_attrs = #fir.var_attrs<intent_in, target>, uniq_name = "_QFdevice_addr_defaultEx"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  SUBROUTINE device_addr_default(x)
    INTEGER, TARGET, INTENT(IN)    :: x(:)
    !$omp target data use_device_addr (x)
    !$omp end target data
  END SUBROUTINE

! Goal: check if we take into account device clause
!CHECK: func.func @{{.*}}device_addr_device_2(
!CHECK: omp.target_data device(%[[DEVICE_ID_CONST:.*]] : i32) use_device_addr(%{{.*}} -> %{{.*}} : !fir.ref<!fir.array<?xi32>>)
!CHECK: %[[DEVICE_ID_CONST_CONV:.*]] = fir.convert %c2_i32 : (i32) -> i64
!CHECK: %[[TGT_PTR1:.*]] = fir.call @__tgt_get_mapped_ptr(%[[DEVICE_ID_CONST_CONV]], %[[BASE_PTR:.*]]) : (i64, !fir.llvm_ptr<i8>) -> !fir.llvm_ptr<i8>
  SUBROUTINE device_addr_device_2(x)
    INTEGER, TARGET, INTENT(IN)    :: x(:)
    !$omp target data use_device_addr (x) device(2)
    !$omp end target data
  END SUBROUTINE

