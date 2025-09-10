// Test lower-nontemporal pass
// RUN: fir-opt --lower-nontemporal %s | FileCheck %s

// CHECK-LABEL: func @_QPsimd_with_nontemporal_clause
func.func @_QPsimd_with_nontemporal_clause(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFsimd_with_nontemporal_clauseEa"}
    // CHECK: %[[A_DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEa"} : (!fir.ref<i32>) -> !fir.ref<i32>
    // CHECK: %[[C_DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEc"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %1 {uniq_name = "_QFsimd_with_nontemporal_clauseEa"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFsimd_with_nontemporal_clauseEb"}
    %4 = fir.declare %3 {uniq_name = "_QFsimd_with_nontemporal_clauseEb"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFsimd_with_nontemporal_clauseEc"}
    %6 = fir.declare %5 {uniq_name = "_QFsimd_with_nontemporal_clauseEc"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimd_with_nontemporal_clauseEi"}
    %8 = fir.declare %7 {uniq_name = "_QFsimd_with_nontemporal_clauseEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFsimd_with_nontemporal_clauseEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %10 = fir.load %9 : !fir.ref<i32>
    // CHECK: omp.simd nontemporal(%[[A_DECL]], %[[C_DECL]] : !fir.ref<i32>, !fir.ref<i32>) {
    // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
    omp.simd nontemporal(%2, %6 : !fir.ref<i32>, !fir.ref<i32>) {
      omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%10) inclusive step (%c1_i32) {
        %11 = fir.declare %8 {uniq_name = "_QFsimd_with_nontemporal_clauseEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
        fir.store %arg2 to %11 : !fir.ref<i32>
        // CHECK:  %[[LOAD:.*]] = fir.load %[[A_DECL]] {nontemporal} : !fir.ref<i32>
        %12 = fir.load %2 : !fir.ref<i32>
        // CHECK:  %[[LOAD1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
        %13 = fir.load %4 : !fir.ref<i32>
        %14 = arith.addi %12, %13 : i32
        // CHECK: %[[ADD_VAL:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
        // CHECK: fir.store %[[ADD_VAL]] to %[[C_DECL]] {nontemporal} : !fir.ref<i32>
        fir.store %14 to %6 : !fir.ref<i32>
        omp.yield
      }
    }
    return
  }

//  CHECK-LABEL:  func.func @_QPsimd_nontemporal_allocatable
func.func @_QPsimd_nontemporal_allocatable(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "y"}) {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c100_i32 = arith.constant 100 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimd_nontemporal_allocatableEi"}
    %2 = fir.declare %1 {uniq_name = "_QFsimd_nontemporal_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    // CHECK:  %[[X_DECL:.*]] = fir.declare %{{.*}} dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, 
    // CHECK-SAME: uniq_name = "_QFsimd_nontemporal_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %3 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsimd_nontemporal_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %4 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFsimd_nontemporal_allocatableEy"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %5 = fir.convert %c100_i32 : (i32) -> index
    %6 = arith.cmpi sgt, %5, %c0 : index
    %7 = arith.select %6, %5, %c0 : index
    %8 = fir.allocmem !fir.array<?xi32>, %7 {fir.must_be_heap = true, uniq_name = "_QFsimd_nontemporal_allocatableEx.alloc"}
    %9 = fir.shape %7 : (index) -> !fir.shape<1>
    %10 = fir.embox %8(%9) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %10 to %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    // CHECK:  omp.simd nontemporal(%[[X_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
    // CHECK:   omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
    omp.simd nontemporal(%3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.loop_nest (%arg3) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32) {
        %16 = fir.declare %2 {uniq_name = "_QFsimd_nontemporal_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
        fir.store %arg3 to %16 : !fir.ref<i32>
        // CHECK:  %[[VAL1:.*]] = fir.load %[[X_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
        %17 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
        // CHECK: %[[VAL2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
        %18 = fir.load %16 : !fir.ref<i32>
        %19 = fir.convert %18 : (i32) -> i64
        // CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[VAL1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
        %20 = fir.box_addr %17 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
        %c0_0 = arith.constant 0 : index
        %21:3 = fir.box_dims %17, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
        %22 = fir.shape_shift %21#0, %21#1 : (index, index) -> !fir.shapeshift<1>
        // CHECK:  %[[ARR_COOR:.*]] = fir.array_coor %[[BOX_ADDR]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, i64) -> !fir.ref<i32>
        %23 = fir.array_coor %20(%22) %19 : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, i64) -> !fir.ref<i32>
        // CHECK:  %[[VAL3:.*]] = fir.load %[[ARR_COOR]] {nontemporal} : !fir.ref<i32>
        %24 = fir.load %23 : !fir.ref<i32>
        %25 = fir.load %4 : !fir.ref<i32>
        %26 = arith.addi %24, %25 : i32
        %27 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
        %28 = fir.load %16 : !fir.ref<i32>
        %29 = fir.convert %28 : (i32) -> i64
        %30 = fir.box_addr %27 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
        %c0_1 = arith.constant 0 : index
        %31:3 = fir.box_dims %27, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
        %32 = fir.shape_shift %31#0, %31#1 : (index, index) -> !fir.shapeshift<1>
        %33 = fir.array_coor %30(%32) %29 : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, i64) -> !fir.ref<i32>
        // CHECK:  fir.store %{{.*}} to %{{.*}} {nontemporal} : !fir.ref<i32>
        fir.store %26 to %33 : !fir.ref<i32>
        omp.yield
      }
    }
    %11 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %12 = fir.box_addr %11 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    fir.freemem %12 : !fir.heap<!fir.array<?xi32>>
    %13 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
    %14 = fir.shape %c0 : (index) -> !fir.shape<1>
    %15 = fir.embox %13(%14) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %15 to %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    return
  }

