// Test lower-nontemporal pass
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s --check-prefixes=CHECK-LABEL,CHECK

// CHECK-LABEL:  llvm.func @_QPtest() 
// CHECK:    %[[CONST_VAL:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[VAL1:.*]] = llvm.alloca %[[CONST_VAL]] x i32 {bindc_name = "n"} : (i64) -> !llvm.ptr
// CHECK:    %[[CONST_VAL1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[VAL2:.*]] = llvm.alloca %[[CONST_VAL1]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CHECK:    %[[CONST_VAL2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[VAL3:.*]] = llvm.alloca %[[CONST_VAL2]] x i32 {bindc_name = "c"} : (i64) -> !llvm.ptr
// CHECK:    %[[CONST_VAL3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[VAL4:.*]] = llvm.alloca %[[CONST_VAL3]] x i32 {bindc_name = "b"} : (i64) -> !llvm.ptr
// CHECK:    %[[CONST_VAL4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[VAL5:.*]] = llvm.alloca %[[CONST_VAL4]] x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
// CHECK:    %[[CONST_VAL5:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:    %[[VAL6:.*]] = llvm.load %[[VAL1]] : !llvm.ptr -> i32
// CHECK:    omp.simd nontemporal(%[[VAL5]], %[[VAL3]] : !llvm.ptr, !llvm.ptr) private(@_QFtestEi_private_i32 %[[VAL2]] -> %arg0 : !llvm.ptr) {
// CHECK:      omp.loop_nest (%{{.*}}) : i32 = (%[[CONST_VAL5]]) to (%[[VAL6]]) inclusive step (%[[CONST_VAL5]]) {
// CHECK:        llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
// CHECK:        %[[VAL8:.*]] = llvm.load %[[VAL5]] {nontemporal} : !llvm.ptr -> i32
// CHECK:        %[[VAL9:.*]] = llvm.load %[[VAL4]] : !llvm.ptr -> i32
// CHECK:        %[[VAL10:.*]] = llvm.add %[[VAL8]], %[[VAL9]] : i32
// CHECK:        llvm.store %[[VAL10]], %[[VAL3]] {nontemporal} : i32, !llvm.ptr
// CHECK:        omp.yield
// CHECK:      }
// CHECK:    }

 func.func @_QPtest() {
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFtestEa"}
    %1 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFtestEb"}
    %2 = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFtestEc"}
    %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
    %4 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFtestEn"}
    %5 = fir.load %4 : !fir.ref<i32>
    omp.simd nontemporal(%0, %2 : !fir.ref<i32>, !fir.ref<i32>) private(@_QFtestEi_private_i32 %3 -> %arg0 : !fir.ref<i32>) {
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%5) inclusive step (%c1_i32) {
        fir.store %arg1 to %arg0 : !fir.ref<i32>
        %6 = fir.load %0 {nontemporal}: !fir.ref<i32>
        %7 = fir.load %1 : !fir.ref<i32>
        %8 = arith.addi %6, %7 : i32
        fir.store %8 to %2 {nontemporal} : !fir.ref<i32>
        omp.yield
      }
    }
    return
  }

// CHECK-LABEL:  llvm.func @_QPsimd_nontemporal_allocatable
// CHECK:    %[[CONST_VAL:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[ALLOCA2:.*]] = llvm.alloca %[[CONST_VAL]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CHECK:    %[[IDX_VAL:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:    %[[CONST_VAL1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:    %[[END_IDX:.*]] = llvm.mlir.constant(100 : i32) : i32
// CHECK:    omp.simd nontemporal(%[[ARG0:.*]] : !llvm.ptr) private(@_QFsimd_nontemporal_allocatableEi_private_i32 %[[ALLOCA2]] -> %[[ARG2:.*]] : !llvm.ptr) {
// CHECK:      omp.loop_nest (%[[ARG3:.*]]) : i32 = (%[[IDX_VAL]]) to (%[[END_IDX]]) inclusive step (%[[IDX_VAL]]) {
// CHECK:        llvm.store %[[ARG3]], %[[ARG2]] : i32, !llvm.ptr
// CHECK:        %[[CONST_VAL2:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK:        "llvm.intr.memcpy"(%[[ALLOCA1:.*]], %[[ARG0]], %[[CONST_VAL2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:        %[[VAL1:.*]] = llvm.load %[[ARG2]] : !llvm.ptr -> i32
// CHECK:        %[[VAL2:.*]] = llvm.sext %[[VAL1]] : i32 to i64
// CHECK:        %[[VAL3:.*]] = llvm.getelementptr %[[ALLOCA1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK:        %[[VAL4:.*]] = llvm.load %[[VAL3]] : !llvm.ptr -> !llvm.ptr
// CHECK:        %[[VAL5:.*]] = llvm.getelementptr %[[ALLOCA1]][0, 7, %[[CONST_VAL1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK:        %[[VAL6:.*]] = llvm.load %[[VAL5]] : !llvm.ptr -> i64
// CHECK:        %[[VAL7:.*]] = llvm.getelementptr %[[ALLOCA1]][0, 7, %[[CONST_VAL1]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK:        %[[VAL8:.*]] = llvm.load %[[VAL7]] : !llvm.ptr -> i64
// CHECK:        %[[VAL10:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:        %[[VAL11:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:        %[[VAL12:.*]] = llvm.sub %[[VAL2]], %[[VAL6]] overflow<nsw> : i64
// CHECK:        %[[VAL13:.*]] = llvm.mul %[[VAL12]], %[[VAL10]] overflow<nsw> : i64
// CHECK:        %[[VAL14:.*]] = llvm.mul %[[VAL13]], %[[VAL10]] overflow<nsw> : i64
// CHECK:        %[[VAL15:.*]] = llvm.add %[[VAL14]], %[[VAL11]] overflow<nsw> : i64
// CHECK:        %[[VAL16:.*]] = llvm.mul %[[VAL10]], %[[VAL8]] overflow<nsw> : i64
// CHECK:        %[[VAL17:.*]] = llvm.getelementptr %[[VAL4]][%[[VAL15]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:        %[[VAL18:.*]] = llvm.load %[[VAL17]] {nontemporal} : !llvm.ptr -> i32
// CHECK:        %[[VAL19:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:        %[[VAL20:.*]] = llvm.add %[[VAL18]], %[[VAL19]] : i32
// CHECK:        llvm.store %[[VAL20]], %[[VAL17]] {nontemporal} : i32, !llvm.ptr
// CHECK:        omp.yield
// CHECK:      }
// CHECK:    }
// CHECK:    llvm.return

  func.func @_QPsimd_nontemporal_allocatable(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "y"}) {
   %c100 = arith.constant 100 : index
   %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c100_i32 = arith.constant 100 : i32
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimd_nontemporal_allocatableEi"}
    %1 = fir.allocmem !fir.array<?xi32>, %c100 {fir.must_be_heap = true, uniq_name = "_QFsimd_nontemporal_allocatableEx.alloc"}
    %2 = fircg.ext_embox %1(%c100) : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    omp.simd nontemporal(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) private(@_QFsimd_nontemporal_allocatableEi_private_i32 %0 -> %arg2 : !fir.ref<i32>) {
      omp.loop_nest (%arg3) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32) {
        fir.store %arg3 to %arg2 : !fir.ref<i32>
        %7 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
        %8 = fir.load %arg2 : !fir.ref<i32>
        %9 = fir.convert %8 : (i32) -> i64
        %10 = fir.box_addr %7 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
        %11:3 = fir.box_dims %7, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
        %12 = fircg.ext_array_coor %10(%11#1) origin %11#0<%9> : (!fir.heap<!fir.array<?xi32>>, index, index, i64) -> !fir.ref<i32>
        %13 = fir.load %12 {nontemporal} : !fir.ref<i32> 
        %14 = fir.load %arg1 : !fir.ref<i32>
        %15 = arith.addi %13, %14 : i32
        fir.store %15 to %12 {nontemporal} : !fir.ref<i32>
        omp.yield
      }
    }
    return
  }
