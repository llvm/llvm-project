// RUN: fir-opt %s -offload-livein-value-canonicalization -split-input-file | FileCheck %s

// -----

// Test fir.shape sinking
func.func private @use_box(!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()

func.func @test_firshape_sink() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  acc.serial {
    %zeroaddr = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %box = fir.embox %zeroaddr(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.call @use_box(%box) : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_firshape_sink
// CHECK: acc.serial {
// CHECK:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[SHAPE:.*]] = fir.shape %[[C10]]
// CHECK:   fir.embox {{.*}}(%[[SHAPE]])

// -----

// Test fir.shape rematerialization
func.func private @use_box(!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()

func.func @test_firshape_rematerialize() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %zeroaddr = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %box = fir.embox %zeroaddr(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.call @use_box(%box) : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()
  acc.serial {
    %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.ref<!fir.array<?xf32>>
    %box2 = fir.embox %addr(%shape) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.call @use_box(%box2) : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_firshape_rematerialize
// CHECK: %[[SHAPE_OUTER:.*]] = fir.shape
// CHECK: fir.embox {{.*}}(%[[SHAPE_OUTER]])
// CHECK: acc.serial {
// CHECK:   %[[SHAPE_INNER:.*]] = fir.shape
// CHECK:   fir.embox {{.*}}(%[[SHAPE_INNER]])

// -----

// Test fir.shape_shift sinking
func.func private @use_box(!fir.box<!fir.array<?xf32>>) -> ()

func.func @test_shapeshift_sink(%arg0: !fir.ref<!fir.array<?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shapeshift = fir.shape_shift %c1, %c10 : (index, index) -> !fir.shapeshift<1>
  acc.serial {
    %box = fir.embox %arg0(%shapeshift) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
    fir.call @use_box(%box) : (!fir.box<!fir.array<?xf32>>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_shapeshift_sink
// CHECK: acc.serial {
// CHECK:   fir.shape_shift
// CHECK:   fir.embox

// -----

// Test fir.field_index sinking
func.func private @use_ref(!fir.ref<f32>) -> ()

func.func @test_fieldindex_sink() {
  %var = fir.alloca !fir.type<_QTmytype{field:f32}>
  %fieldidx = fir.field_index field, !fir.type<_QTmytype{field:f32}>
  acc.serial {
    %coor = fir.coordinate_of %var, %fieldidx : (!fir.ref<!fir.type<_QTmytype{field:f32}>>, !fir.field) -> !fir.ref<f32>
    fir.call @use_ref(%coor) : (!fir.ref<f32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_fieldindex_sink
// CHECK: acc.serial {
// CHECK:   %[[FIELD:.*]] = fir.field_index field
// CHECK:   fir.coordinate_of {{.*}}, %[[FIELD]]

// -----

// Test fir.address_of with acc.declare sinking
fir.global @global_with_declare {acc.declare = #acc.declare<dataClause = acc_copyin>} : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

func.func private @use_ref(!fir.ref<f32>) -> ()

func.func @test_address_of_with_declare_sink() {
  %addr = fir.address_of(@global_with_declare) : !fir.ref<f32>
  acc.serial {
    fir.call @use_ref(%addr) : (!fir.ref<f32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_address_of_with_declare_sink
// CHECK: acc.serial {
// CHECK:   %[[ADDR:.*]] = fir.address_of(@global_with_declare)
// CHECK:   fir.call @use_ref(%[[ADDR]])

// -----

// Test fir.address_of with constant global sinking
fir.global @global_constant constant : f32 {
  %0 = arith.constant 42.0 : f32
  fir.has_value %0 : f32
}

func.func private @use_ref(!fir.ref<f32>) -> ()

func.func @test_address_of_constant_global_sink() {
  %addr = fir.address_of(@global_constant) : !fir.ref<f32>
  acc.serial {
    fir.call @use_ref(%addr) : (!fir.ref<f32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_address_of_constant_global_sink
// CHECK: acc.serial {
// CHECK:   %[[ADDR:.*]] = fir.address_of(@global_constant)
// CHECK:   fir.call @use_ref(%[[ADDR]])

// -----

// Test fir.address_of with fir.convert tracing (ViewLikeOpInterface)
fir.global @global_for_convert {acc.declare = #acc.declare<dataClause = acc_copyin>} : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

func.func private @use_ptr(!fir.ptr<f32>) -> ()

func.func @test_address_of_with_convert_sink() {
  %addr = fir.address_of(@global_for_convert) : !fir.ref<f32>
  %converted = fir.convert %addr : (!fir.ref<f32>) -> !fir.ptr<f32>
  acc.serial {
    fir.call @use_ptr(%converted) : (!fir.ptr<f32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_address_of_with_convert_sink
// CHECK: acc.serial {
// CHECK:   fir.address_of(@global_for_convert)
// CHECK:   fir.convert

// -----

// Test fir.declare with PartialEntityAccessOpInterface tracing
fir.global @global_for_declare {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.array<10xf32> {
  %0 = fir.zero_bits !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func private @use_ref(!fir.ref<!fir.array<10xf32>>) -> ()

func.func @test_address_of_through_declare_sink() {
  %c10 = arith.constant 10 : index
  %addr = fir.address_of(@global_for_declare) : !fir.ref<!fir.array<10xf32>>
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %decl = fir.declare %addr(%shape) {uniq_name = "global"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  acc.serial {
    fir.call @use_ref(%decl) : (!fir.ref<!fir.array<10xf32>>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_address_of_through_declare_sink
// CHECK: acc.serial {
// CHECK-DAG:   fir.address_of(@global_for_declare)
// CHECK-DAG:   fir.shape
// CHECK:   fir.declare

// -----

// Test 2D shape sinking for array operations
func.func @test_2d_shape_sink() {
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca !fir.array<3x32xf32>
  %shape = fir.shape %c3, %c32 : (index, index) -> !fir.shape<2>
  %decl = fir.declare %0(%shape) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<3x32xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<3x32xf32>>
  acc.serial {
    %coor = fir.array_coor %decl(%shape) %c1, %c1 : (!fir.ref<!fir.array<3x32xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_2d_shape_sink
// CHECK: acc.serial {
// CHECK:   fir.shape
// CHECK:   fir.array_coor

// -----

// Test acc.bounds sinking with FIR types
func.func @test_accbounds_sink_fir() {
  %c1 = arith.constant 1 : index
  %bounds = acc.bounds upperbound(%c1 : index)
  acc.serial {
    %local = fir.alloca i32
    %priv = acc.private varPtr(%local : !fir.ref<i32>) bounds(%bounds) -> !fir.ref<i32>
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_accbounds_sink_fir
// CHECK: acc.serial {
// CHECK:   acc.bounds

// -----

// Test acc.bounds rematerialization with FIR types
func.func @test_accbounds_rematerialize_fir() {
  %c1 = arith.constant 1 : index
  %bounds = acc.bounds upperbound(%c1 : index)
  %local = fir.alloca i32
  %priv = acc.private varPtr(%local : !fir.ref<i32>) bounds(%bounds) -> !fir.ref<i32>
  acc.serial {
    %priv2 = acc.private varPtr(%local : !fir.ref<i32>) bounds(%bounds) -> !fir.ref<i32>
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_accbounds_rematerialize_fir
// CHECK: acc.bounds
// CHECK: acc.serial {
// CHECK:   acc.bounds
