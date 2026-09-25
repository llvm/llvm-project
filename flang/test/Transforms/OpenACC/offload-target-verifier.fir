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
  %alloca = fir.alloca !fir.array<10xf32>
  // expected-warning @below {{1 illegal live-in value(s)}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %coor = fir.coordinate_of %alloca, %arg0 : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    %load = fir.load %coor : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Test cuf.kernel region with host scalar live-in - should pass
// (scalars are passed by value)
func.func @test_cuf_kernel_scalar() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = fir.alloca f32
  // expected-remark @below {{passed validity check}}
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

// -----

// Test fir.use_stmt inside offload region referencing undeclared global
fir.global @_QMmod1Evar1 : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

func.func @test_use_stmt_no_declare() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.use_stmt "mod1" only_symbols[[@_QMmod1Evar1]]
    acc.yield
  }
  return
}

// -----

// Test fir.use_stmt inside offload region referencing declared global
fir.global @_QMmod2Evar2 {acc.declare = #acc.declare<dataClause = acc_create>} : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

func.func @test_use_stmt_with_declare() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.use_stmt "mod2" only_symbols[[@_QMmod2Evar2]]
    acc.yield
  }
  return
}

// -----

// A fir.use_stmt does not require the referenced symbol to have a definition.
func.func @test_use_stmt_undefined_symbol() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.use_stmt "mod3" only_symbols[[@_QMmod3Evar3]]
    acc.yield
  }
  return
}

// -----

// A fir.use_stmt with only a module name has no symbol use to validate.
func.func @test_use_stmt_module_name() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.use_stmt "mod4"
    acc.yield
  }
  return
}

// -----

func.func private @_FortranARuntime() attributes {fir.runtime}

func.func @test_fir_runtime_symbol() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.call @_FortranARuntime() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func private @cuf_device_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<device>}

func.func @test_cuf_device_proc() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.call @cuf_device_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func private @cuf_global_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<global>}

func.func @test_cuf_global_proc() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.call @cuf_global_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func private @cuf_grid_global_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<grid_global>}

func.func @test_cuf_grid_global_proc() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.call @cuf_grid_global_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func private @cuf_host_device_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>}

func.func @test_cuf_host_device_proc() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    fir.call @cuf_host_device_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func private @cuf_host_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<host>}

func.func @test_cuf_host_proc() {
  // expected-warning @below {{illegal symbol(s): cuf_host_proc}}
  acc.serial {
    fir.call @cuf_host_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

func.func @test_undefined_symbol() {
  // expected-warning @below {{illegal symbol(s): undefined_proc}}
  acc.serial {
    fir.call @undefined_proc() : () -> ()
    acc.yield
  }
  return
}

// -----

// A length-one FIR character is passed by value.
func.func @test_fir_character_scalar() {
  %alloca = fir.alloca !fir.char<1,1>
  %livein = fir.load %alloca : !fir.ref<!fir.char<1,1>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %sink = fir.alloca !fir.char<1,1>
    fir.store %livein to %sink : !fir.ref<!fir.char<1,1>>
    acc.yield
  }
  return
}

// -----

// A multi-character FIR value is not passed as a scalar.
func.func @test_fir_character_nonscalar() {
  %alloca = fir.alloca !fir.char<1,2>
  // expected-note @below {{value}}
  %livein = fir.load %alloca : !fir.ref<!fir.char<1,2>>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %sink = fir.alloca !fir.char<1,2>
    fir.store %livein to %sink : !fir.ref<!fir.char<1,2>>
    acc.yield
  }
  return
}

// -----

// A complex scalar is passed by value to a CUF kernel.
func.func @test_cuf_kernel_complex_scalar() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = fir.alloca complex<f32>
  // expected-remark @below {{passed validity check}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<complex<f32>>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// A record composed of scalar fields is passed by value to a CUF kernel.
func.func @test_cuf_kernel_record_scalar() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = fir.alloca !fir.type<_QFTpoint{x:f32,y:i32}>
  // expected-remark @below {{passed validity check}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<!fir.type<_QFTpoint{x:f32,y:i32}>>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Trivially-copyable records may be nested.
func.func @test_cuf_kernel_nested_record_scalar() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %alloca = fir.alloca !fir.type<_QFTouter{inner:!fir.type<_QFTinner{x:f32}>,flag:!fir.logical<4>}>
  // expected-remark @below {{passed validity check}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<!fir.type<_QFTouter{inner:!fir.type<_QFTinner{x:f32}>,flag:!fir.logical<4>}>>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// A record containing an array is not passed by value.
func.func @test_cuf_kernel_nonscalar_record() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  // expected-note @below {{value}}
  %alloca = fir.alloca !fir.type<_QFTwitharray{values:!fir.array<10xf32>}>
  // expected-warning @below {{1 illegal live-in value(s)}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<!fir.type<_QFTwitharray{values:!fir.array<10xf32>}>>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// A nested record is not passable by value if any nested field is not.
func.func @test_cuf_kernel_nested_nonscalar_record() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  // expected-note @below {{value}}
  %alloca = fir.alloca !fir.type<_QFTbadouter{inner:!fir.type<_QFTbadinner{values:!fir.array<10xf32>}>}>
  // expected-warning @below {{1 illegal live-in value(s)}}
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %load = fir.load %alloca : !fir.ref<!fir.type<_QFTbadouter{inner:!fir.type<_QFTbadinner{values:!fir.array<10xf32>}>}>>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// CUF-originated compute regions use the same scalar live-in rules.
func.func @test_cuf_compute_region_record_scalar() {
  %alloca = fir.alloca !fir.type<_QFTcompute_point{x:f32}>
  %decl = fir.declare %alloca {uniq_name = "_QFEpoint"} : (!fir.ref<!fir.type<_QFTcompute_point{x:f32}>>) -> !fir.ref<!fir.type<_QFTcompute_point{x:f32}>>
  %blocks = acc.par_width par_dim(#acc.par_dim<block_x>)
  %threads = acc.par_width par_dim(#acc.par_dim<thread_x>)
  // expected-remark @below {{passed validity check}}
  acc.compute_region launch(%arg0 = %blocks, %arg1 = %threads) ins(%arg2 = %decl) : (!fir.ref<!fir.type<_QFTcompute_point{x:f32}>>) {
    %load = fir.load %arg2 : !fir.ref<!fir.type<_QFTcompute_point{x:f32}>>
    acc.yield
  } <{kernel_func_name = @cuf_compute_region_record_scalar, kernel_module_name = @cuda_device_mod, origin = "cuf.kernel"}>
  return
}

// -----

// A view of an ACC data-entry value remains valid.
func.func @test_acc_data_entry_view() {
  %alloca = fir.alloca f32
  %copyin = acc.copyin varPtr(%alloca : !fir.ref<f32>) -> !fir.ref<f32>
  %view = fir.convert %copyin : (!fir.ref<f32>) -> !fir.ptr<f32>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %load = fir.load %view : !fir.ptr<f32>
    acc.yield
  }
  return
}
