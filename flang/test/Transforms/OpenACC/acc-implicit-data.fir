// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// -----

func.func @test_fir_scalar_in_serial() {
  %livein = fir.alloca i64 {bindc_name = "scalarvar"}
  acc.serial {
    %load = fir.load %livein : !fir.ref<i64>
    acc.yield
  }
  return
}

// CHECK: acc.firstprivate varPtr({{.*}} : !fir.ref<i64>) -> !fir.ref<i64> {implicit = true, name = "scalarvar"}

// -----

func.func @test_fir_scalar_in_parallel() {
  %livein = fir.alloca f32 {bindc_name = "scalarvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<f32>
    acc.yield
  }
  return
}

// CHECK: acc.firstprivate varPtr({{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {implicit = true, name = "scalarvar"}

// -----

func.func @test_fir_scalar_in_kernels() {
  %livein = fir.alloca f64 {bindc_name = "scalarvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<f64>
    acc.terminator
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<f64>) -> !fir.ref<f64> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "scalarvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<f64>) to varPtr({{.*}} : !fir.ref<f64>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "scalarvar"}

// -----

func.func @test_fir_scalar_in_parallel_defaultnone() {
  %livein = fir.alloca f32 {bindc_name = "scalarvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<f32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue none>}
  return
}

// CHECK-NOT: acc.firstprivate

// -----

func.func @test_fir_scalar_in_kernels_defaultnone() {
  %livein = fir.alloca f64 {bindc_name = "scalarvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<f64>
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue none>}
  return
}

// CHECK-NOT: acc.copyin

// -----

func.func @test_fir_derivedtype_in_parallel() {
  %livein = fir.alloca !fir.type<_QFTaggr{field:f32}> {bindc_name = "aggrvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<!fir.type<_QFTaggr{field:f32}>>
    acc.yield
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) -> !fir.ref<!fir.type<_QFTaggr{field:f32}>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "aggrvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) to varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "aggrvar"}

// -----

func.func @test_fir_derivedtype_in_kernels() {
  %livein = fir.alloca !fir.type<_QFTaggr{field:f32}> {bindc_name = "aggrvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<!fir.type<_QFTaggr{field:f32}>>
    acc.terminator
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) -> !fir.ref<!fir.type<_QFTaggr{field:f32}>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "aggrvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) to varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "aggrvar"}

// -----

func.func @test_fir_array_in_parallel() {
  %livein = fir.alloca !fir.array<10xf32> {bindc_name = "arrayvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<!fir.array<10xf32>>
    acc.yield
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "arrayvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10xf32>>) to varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "arrayvar"}

// -----

func.func @test_fir_array_in_kernels() {
  %livein = fir.alloca !fir.array<10xf32> {bindc_name = "arrayvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<!fir.array<10xf32>>
    acc.terminator
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "arrayvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10xf32>>) to varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "arrayvar"}

// -----

func.func @test_fir_derivedtype_in_parallel_defaultpresent() {
  %livein = fir.alloca !fir.type<_QFTaggr{field:f32}> {bindc_name = "aggrvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<!fir.type<_QFTaggr{field:f32}>>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: %[[PRESENT:.*]] = acc.present varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) -> !fir.ref<!fir.type<_QFTaggr{field:f32}>> {implicit = true, name = "aggrvar"}
// CHECK: acc.delete accPtr(%[[PRESENT]] : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) {dataClause = #acc<data_clause acc_present>, implicit = true, name = "aggrvar"}

// -----

func.func @test_fir_derivedtype_in_kernels_defaultpresent() {
  %livein = fir.alloca !fir.type<_QFTaggr{field:f32}> {bindc_name = "aggrvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<!fir.type<_QFTaggr{field:f32}>>
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: %[[PRESENT:.*]] = acc.present varPtr({{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) -> !fir.ref<!fir.type<_QFTaggr{field:f32}>> {implicit = true, name = "aggrvar"}
// CHECK: acc.delete accPtr(%[[PRESENT]] : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) {dataClause = #acc<data_clause acc_present>, implicit = true, name = "aggrvar"}

// -----

func.func @test_fir_array_in_parallel_defaultpresent() {
  %livein = fir.alloca !fir.array<10xf32> {bindc_name = "arrayvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<!fir.array<10xf32>>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: %[[PRESENT:.*]] = acc.present varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {implicit = true, name = "arrayvar"}
// CHECK: acc.delete accPtr(%[[PRESENT]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_present>, implicit = true, name = "arrayvar"}

// -----

func.func @test_fir_array_in_kernels_defaultpresent() {
  %livein = fir.alloca !fir.array<10xf32> {bindc_name = "arrayvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<!fir.array<10xf32>>
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: %[[PRESENT:.*]] = acc.present varPtr({{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {implicit = true, name = "arrayvar"}
// CHECK: acc.delete accPtr(%[[PRESENT]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_present>, implicit = true, name = "arrayvar"}

// -----

func.func @test_fir_scalar_in_parallel_defaultpresent() {
  %livein = fir.alloca f32 {bindc_name = "scalarvar"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<f32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: acc.firstprivate varPtr({{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {implicit = true, name = "scalarvar"}

// -----

func.func @test_fir_scalar_in_kernels_defaultpresent() {
  %livein = fir.alloca f64 {bindc_name = "scalarvar"}
  acc.kernels {
    %load = fir.load %livein : !fir.ref<f64>
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<f64>) -> !fir.ref<f64> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "scalarvar"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<f64>) to varPtr({{.*}} : !fir.ref<f64>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "scalarvar"}

// -----

func.func @test_fir_box_ref() {
  %livein = fir.alloca !fir.box<!fir.array<?xi32>> {bindc_name = "descriptor"}
  acc.parallel {
    %load = fir.load %livein : !fir.ref<!fir.box<!fir.array<?xi32>>>
    acc.yield
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.array<?xi32>>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "descriptor"}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) to varPtr({{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "descriptor"}

// -----

func.func @test_fir_box_val() {
  %desc = fir.alloca !fir.box<!fir.array<?xi32>> {bindc_name = "descriptor"}
  %livein = fir.load %desc : !fir.ref<!fir.box<!fir.array<?xi32>>>
  acc.parallel {
    %addr = fir.box_addr %livein : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    acc.yield
  }
  return
}

// CHECK: %[[COPYIN:.*]] = acc.copyin var({{.*}} : !fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "descriptor"}
// CHECK: acc.copyout accVar(%[[COPYIN]] : !fir.box<!fir.array<?xi32>>) to var({{.*}} : !fir.box<!fir.array<?xi32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "descriptor"}


// -----

// This test has an explicit data clause for the box - but the pointer held
// inside the box is used in the region instead of the box itself. Test that
// implicit present is actually used.
func.func @test_explicit_box_implicit_ptr() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %arr = fir.alloca !fir.array<10xf32> {bindc_name = "aa"}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %arr_decl = fir.declare %arr(%shape) {uniq_name = "aa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %box = fir.embox %arr_decl(%shape) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  %copyin = acc.copyin var(%box : !fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "aa"}
  acc.serial dataOperands(%copyin : !fir.box<!fir.array<10xf32>>) {
    // Use the pointer, not the box
    %elem = fir.array_coor %arr_decl(%shape) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    acc.yield
  }
  acc.copyout accVar(%copyin : !fir.box<!fir.array<10xf32>>) to var(%box : !fir.box<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "aa"}
  return
}

// CHECK: acc.present varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>){{.*}}-> !fir.ref<!fir.array<10xf32>> {implicit = true, name = "aa"}

// -----

// This test uses an explicit-shape array with no data clause - it also has
// an optimization where the pointer is used instead of the boxed entity.
// It tests that the implicit data pass is able to recover the size despite
// it not being encoded in the FIR type.
// It was generated from the following Fortran source:
//   subroutine array(aa,nn)
//     integer :: nn
//     real :: aa(10:nn)
//     !$acc kernels loop
//     do ii = 10, nn
//       aa(ii) = ii
//     end do
//     !$acc end kernels
//   end subroutine

func.func @_QParray(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "aa"}, %arg1: !fir.ref<i32> {fir.bindc_name = "nn"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10_i64 = arith.constant 10 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFarrayEnn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.convert %c10_i64 : (i64) -> index
  %5 = fir.load %1 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> i64
  %7 = fir.convert %6 : (i64) -> index
  %8 = arith.subi %7, %4 : index
  %9 = arith.addi %8, %c1 : index
  %10 = arith.cmpi sgt, %9, %c0 : index
  %11 = arith.select %10, %9, %c0 : index
  %12 = fir.shape_shift %4, %11 : (index, index) -> !fir.shapeshift<1>
  %13 = fir.declare %arg0(%12) dummy_scope %0 {uniq_name = "_QFarrayEaa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  acc.kernels {
    %elem = fir.array_coor %13(%12) %4 : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// This tries to confirm that the acc.bounds operation is as expected.
// Effectively the extent needs to be max(0, nn), stride needs to be 1,
// adjusted lowerbound is 0, and actual language start index is 10.
// CHECK: %[[NN:.*]] = fir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFarrayEnn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK: %[[C10:.*]] = fir.convert %c10{{.*}} : (i64) -> index
// CHECK: %[[LOADEDNN:.*]] = fir.load %[[NN]] : !fir.ref<i32>
// CHECK: %[[CAST1:.*]] = fir.convert %[[LOADEDNN]] : (i32) -> i64
// CHECK: %[[CAST2:.*]]  = fir.convert %[[CAST1]] : (i64) -> index
// CHECK: %[[SUBI:.*]] = arith.subi %[[CAST2]], %[[C10]] : index
// CHECK: %[[ADDI:.*]] = arith.addi %[[SUBI]], %c1{{.*}} : index
// CHECK: %[[CMPI:.*]] = arith.cmpi sgt, %[[ADDI]], %c0{{.*}} : index
// CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]], %[[ADDI]], %c0{{.*}} : index
// CHECK: %[[BOUNDS:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[SELECT]] : index) stride(%c1{{.*}} : index) startIdx(%[[C10]] : index)
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "aa"}

// -----

// Test to confirm that a copyin clause is not implicitly generated for deviceptr symbol.
func.func @test_deviceptr_no_implicit_copy() {
  %c10 = arith.constant 10 : index
  %arr = fir.alloca !fir.array<10xf64> {bindc_name = "a"}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %arr_box = fir.embox %arr(%shape) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  %devptr = acc.deviceptr var(%arr_box : !fir.box<!fir.array<10xf64>>) -> !fir.box<!fir.array<10xf64>> {name = "a"}
  acc.parallel dataOperands(%devptr : !fir.box<!fir.array<10xf64>>) {
    %elem = fir.box_addr %arr_box : (!fir.box<!fir.array<10xf64>>) -> !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  return
}

// CHECK-NOT: acc.copyin
// CHECK: acc.deviceptr

// -----

// Test that acc.declare with deviceptr doesn't generate implicit copyin
func.func @test_acc_declare_deviceptr() {
  %c10 = arith.constant 10 : index
  %arr = fir.alloca !fir.array<10xf64> {bindc_name = "a"}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %arr_box = fir.embox %arr(%shape) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  %devptr = acc.deviceptr var(%arr_box : !fir.box<!fir.array<10xf64>>) -> !fir.box<!fir.array<10xf64>> {name = "a"}
  %token = acc.declare_enter dataOperands(%devptr : !fir.box<!fir.array<10xf64>>)
  acc.parallel {
    %elem = fir.box_addr %arr_box : (!fir.box<!fir.array<10xf64>>) -> !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  acc.declare_exit token(%token)
  return
}

// CHECK-LABEL: func.func @test_acc_declare_deviceptr
// CHECK: acc.deviceptr
// CHECK-NOT: acc.copyin
// CHECK: acc.deviceptr

