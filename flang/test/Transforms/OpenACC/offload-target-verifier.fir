// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,func.func(offload-target-verifier{soft-check=true}))" --verify-diagnostics -split-input-file

// Test scalar i32 live-in value - should pass (scalars can be passed by value)
func.func @test_scalar_i32() {
  %alloca = fir.alloca i32
  %livein = fir.load %alloca : !fir.ref<i32>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %accalloca = fir.alloca i32
    fir.store %livein to %accalloca : !fir.ref<i32>
    acc.yield
  }
  return
}

// -----

// Test fir.logical scalar - should pass (scalars can be passed by value)
func.func @test_fir_logical_scalar() {
  %alloca = fir.alloca !fir.logical<4>
  %livein = fir.load %alloca : !fir.ref<!fir.logical<4>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %accalloca = fir.alloca !fir.logical<4>
    fir.store %livein to %accalloca : !fir.ref<!fir.logical<4>>
    acc.yield
  }
  return
}

// -----

// Test fir.ref live-in without data clause - should fail
func.func @test_fir_ref() {
  // expected-note @below {{value}}
  %livein = fir.alloca f32
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %load = fir.load %livein : !fir.ref<f32>
    %accalloca = fir.alloca f32
    fir.store %load to %accalloca : !fir.ref<f32>
    acc.yield
  }
  return
}

// -----

// Test fir.ref with copyin data clause - should pass
func.func @test_fir_ref_copyin() {
  %alloca = fir.alloca f32
  %livein = acc.copyin varPtr(%alloca : !fir.ref<f32>) -> !fir.ref<f32>
  // expected-remark @below {{passed validity check}}
  acc.serial dataOperands(%livein : !fir.ref<f32>) {
    %load = fir.load %livein : !fir.ref<f32>
    %accalloca = fir.alloca f32
    fir.store %load to %accalloca : !fir.ref<f32>
    acc.yield
  }
  return
}

// -----

// Test fir.ref with private clause - should pass
acc.private.recipe @privatization_ref_f32 : !fir.ref<f32> init {
^bb0(%arg0: !fir.ref<f32>):
  %0 = fir.alloca f32
  acc.yield %0 : !fir.ref<f32>
}

func.func @test_fir_ref_private() {
  %livein = fir.alloca f32
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %private = acc.private varPtr(%livein : !fir.ref<f32>) recipe(@privatization_ref_f32) -> !fir.ref<f32>
    %load = fir.load %private : !fir.ref<f32>
    %accalloca = fir.alloca f32
    fir.store %load to %accalloca : !fir.ref<f32>
    acc.yield
  }
  return
}

// -----

// Test fir.global without declare attribute - should fail
fir.global @_global_array : !fir.array<10xf32> {
  %0 = fir.zero_bits !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @test_fir_global_no_declare() {
  // expected-warning @below {{illegal symbol(s): _global_array}}
  acc.serial {
    %liveinsym = fir.address_of(@_global_array) : !fir.ref<!fir.array<10xf32>>
    %loaded = fir.load %liveinsym : !fir.ref<!fir.array<10xf32>>
    acc.yield
  }
  return
}

// -----

// Test fir.global with acc.declare attribute - should pass
fir.global @_global_array_declared {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<10xf32> {
  %0 = fir.zero_bits !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @test_fir_global_with_declare() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %liveinsym = fir.address_of(@_global_array_declared) : !fir.ref<!fir.array<10xf32>>
    acc.yield
  }
  return
}

// -----

// Test fir.global with CUDA device attribute - should pass
fir.global @_cuda_global_array {data_attr = #cuf.cuda<device>} : !fir.array<10xf32> {
  %0 = fir.zero_bits !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @test_fir_cuda_global() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %liveinsym = fir.address_of(@_cuda_global_array) : !fir.ref<!fir.array<10xf32>>
    acc.yield
  }
  return
}

// -----

// Test fir.declare with CUDA device attribute - should pass
func.func @test_fir_declare_cuda() {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "cuda_array"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %3 = fir.load %2 : !fir.ref<!fir.array<10xf32>>
    acc.yield
  }
  return
}

// -----

// Test hlfir.declare live-in without data clause - should fail
func.func @test_hlfir_declare(%arg0: !fir.ref<f32> {fir.bindc_name = "var"}) {
  %0 = fir.dummy_scope : !fir.dscope
  // expected-note @below {{value}}
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QEvar"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %cst = arith.constant 1.000000e+00 : f32
    hlfir.assign %cst to %1#0 : f32, !fir.ref<f32>
    acc.yield
  }
  return
}

// -----

// Test acc.parallel region
func.func @test_acc_parallel() {
  // expected-note @below {{value}}
  %alloca = fir.alloca f32
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.parallel {
    %load = fir.load %alloca : !fir.ref<f32>
    acc.yield
  }
  return
}

// -----

// Test acc.kernels region
func.func @test_acc_kernels() {
  // expected-note @below {{value}}
  %alloca = fir.alloca f32
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.kernels {
    %load = fir.load %alloca : !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// Test cuf.kernel region with invalid live-in - should fail
func.func @test_cuf_kernel_invalid() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  // expected-note @below {{value}}
  %alloca = fir.alloca f32
  // expected-warning @below {{1 illegal live-in value(s)}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Test cuf.kernel region with CUDA device scalar - should pass
func.func @test_cuf_kernel_cuda_device() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = fir.alloca f32
  %decl = fir.declare %alloca {data_attr = #cuf.cuda<device>, uniq_name = "cuda_scalar"} : (!fir.ref<f32>) -> !fir.ref<f32>
  // expected-remark @below {{passed validity check}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %decl : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Test that fir.shape live-in to cuf.kernel is illegal
func.func @test_cuf_kernel_shape_illegal() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %alloca = fir.alloca !fir.array<10xf32>
  // expected-note @below {{value}}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %decl = fir.declare %alloca(%shape) {data_attr = #cuf.cuda<device>, uniq_name = "cuda_arr"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  // expected-warning @below {{1 illegal live-in value(s)}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c10 : index) step (%c1 : index) {
    %coor = fir.array_coor %decl(%shape) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %load = fir.load %coor : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Test cuf.kernel region with cuf.alloc device data - should pass
func.func @test_cuf_kernel_cuf_alloc() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = cuf.alloc f32 {data_attr = #cuf.cuda<device>} -> !fir.ref<f32>
  // expected-remark @below {{passed validity check}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Test fir.rebox live-in - should fail (box without data clause)
func.func @test_fir_rebox(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "arr"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QEarr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  // expected-note @below {{value}}
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %c0 = arith.constant 0 : index
    %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
    acc.yield
  }
  return
}

// -----

// Test fir.rebox with CUDA device attribute - should pass
func.func @test_fir_rebox_cuda(%arg0: !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "arr"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QEarr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %c0 = arith.constant 0 : index
    %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
    acc.yield
  }
  return
}

// -----

// Test fir.embox live-in - should fail
func.func @test_fir_embox() {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-note @below {{value}}
  %2 = fir.embox %0(%1) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %c0 = arith.constant 0 : index
    %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<10xf32>>, index) -> (index, index, index)
    acc.yield
  }
  return
}
