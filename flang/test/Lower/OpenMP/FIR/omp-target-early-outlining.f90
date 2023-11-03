!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s 
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-gpu -fopenmp-is-target-device %s -o - | FileCheck %s 

!CHECK: func.func @_QPtarget_function

!CHECK:  func.func @_QPwrite_index_omp_outline_0(%[[ARG0:.*]]: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
!CHECK-NEXT: %[[map_info0:.*]] = omp.map_info var_ptr(%[[ARG0]]{{.*}}
!CHECK-NEXT: omp.target  map_entries(%[[map_info0]]{{.*}} {
!CHECK: %[[CONSTANT_VALUE_10:.*]] = arith.constant 10 : i32
!CHECK: fir.store %[[CONSTANT_VALUE_10]] to %[[ARG0]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: return

!CHECK:  func.func @_QPwrite_index_omp_outline_1(%[[ARG1:.*]]: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
!CHECK-NEXT: %[[map_info1:.*]] = omp.map_info var_ptr(%[[ARG1]]{{.*}}
!CHECK-NEXT: omp.target  map_entries(%[[map_info1]]{{.*}} {
!CHECK: %[[CONSTANT_VALUE_20:.*]] = arith.constant 20 : i32
!CHECK: fir.store %[[CONSTANT_VALUE_20]] to %[[ARG1]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: return


SUBROUTINE WRITE_INDEX(INT_ARRAY)
        INTEGER :: INT_ARRAY(*)
        INTEGER :: NEW_LEN
!$omp target map(from:new_len)
        NEW_LEN = 10
!$omp end target
!$omp target map(from:new_len)
        NEW_LEN = 20
!$omp end target
        do INDEX_ = 1, NEW_LEN
                INT_ARRAY(INDEX_) = INDEX_
        end do
end subroutine WRITE_INDEX

SUBROUTINE TARGET_FUNCTION()
!$omp declare target
END

!CHECK: func.func @_QParray_bounds_omp_outline_0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<!fir.array<10xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QParray_bounds"} {
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[C4:.*]] = arith.constant 4 : index
!CHECK: %[[C1_0:.*]] = arith.constant 1 : index
!CHECK: %[[C1_1:.*]] = arith.constant 1 : index
!CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound(%[[C1]] : index) upper_bound(%[[C4]] : index) stride(%[[C1_1]] : index) start_idx(%[[C1_1]] : index)
!CHECK: %[[ENTRY:.*]] = omp.map_info var_ptr(%[[ARG1]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!CHECK: omp.target   map_entries(%[[ENTRY]] : !fir.ref<!fir.array<10xi32>>) {
!CHECK:  %c2_i32 = arith.constant 2 : i32
!CHECK:  %2 = fir.convert %c2_i32 : (i32) -> index
!CHECK:  %c5_i32 = arith.constant 5 : i32
!CHECK:  %3 = fir.convert %c5_i32 : (i32) -> index
!CHECK:  %c1_2 = arith.constant 1 : index
!CHECK:  %4 = fir.convert %2 : (index) -> i32
!CHECK:  %5:2 = fir.do_loop %arg2 = %2 to %3 step %c1_2 iter_args(%arg3 = %4) -> (index, i32) {
!CHECK:    fir.store %arg3 to %[[ARG0]] : !fir.ref<i32>
!CHECK:    %6 = fir.load %[[ARG0]] : !fir.ref<i32>
!CHECK:    %7 = fir.load %[[ARG0]] : !fir.ref<i32>
!CHECK:    %8 = fir.convert %7 : (i32) -> i64
!CHECK:    %c1_i64 = arith.constant 1 : i64
!CHECK:    %9 = arith.subi %8, %c1_i64 : i64
!CHECK:    %10 = fir.coordinate_of %[[ARG1]], %9 : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
!CHECK:    fir.store %6 to %10 : !fir.ref<i32>
!CHECK:    %11 = arith.addi %arg2, %c1_2 : index
!CHECK:    %12 = fir.convert %c1_2 : (index) -> i32
!CHECK:    %13 = fir.load %[[ARG0]] : !fir.ref<i32>
!CHECK:    %14 = arith.addi %13, %12 : i32
!CHECK:    fir.result %11, %14 : index, i32
!CHECK:  }
!CHECK: fir.store %5#1 to %[[ARG0]] : !fir.ref<i32>
!CHECK:  omp.terminator
!CHECK: }
!CHECK:return
!CHECK:}

SUBROUTINE ARRAY_BOUNDS()
        INTEGER :: sp_write(10) = (/0,0,0,0,0,0,0,0,0,0/)
!$omp target map(tofrom:sp_write(2:5))
        do i = 2, 5
                sp_write(i) = i
        end do
!$omp end target
end subroutine ARRAY_BOUNDS
