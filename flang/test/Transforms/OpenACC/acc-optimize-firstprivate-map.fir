// RUN: fir-opt %s --acc-optimize-firstprivate-map -split-input-file | FileCheck %s

// Test: Integer variable - should optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_trivial_scalar_hoist
func.func @test_trivial_scalar_hoist() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_trivial_scalarEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel
  acc.parallel {
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: fir.call @use_i32(%[[LOAD]])
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Inside offload region - should optimize by getting rid of firstprivate_map

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_inside_offload_region
func.func @test_inside_offload_region() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_inside_offloadEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %copyin = acc.copyin varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL]] : !fir.ref<i32>) -> !fir.ref<i32>
  acc.parallel dataOperands(%copyin : !fir.ref<i32>) {
    %fpmap = acc.firstprivate_map varPtr(%copyin : !fir.ref<i32>) -> !fir.ref<i32>
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: acc.parallel dataOperands(%[[COPYIN]] : !fir.ref<i32>)
    // CHECK-NOT: acc.firstprivate_map
    // CHECK: %[[LOAD:.*]] = fir.load %[[COPYIN]] : !fir.ref<i32>
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Local alloca inside offload region - should optimize (erase)

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_local_alloca_inside_offload
func.func @test_local_alloca_inside_offload() {
  acc.parallel {
    %local = fir.alloca i32 {bindc_name = "local_var"}
    %decl = fir.declare %local uniq_name("_QFtest_local_allocaElocal_var") : (!fir.ref<i32>) -> !fir.ref<i32>
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: acc.parallel
    // CHECK: %[[LOCAL:.*]] = fir.alloca i32
    // CHECK: %[[DECL:.*]] = fir.declare %[[LOCAL]]
    // CHECK-NOT: acc.firstprivate_map
    // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: firstprivate_map with acc.private input - should optimize (erase)

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_private_input
func.func @test_private_input() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_private_inputEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %private = acc.private varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[DECL]] : !fir.ref<i32>) -> !fir.ref<i32>
  acc.parallel private(%private : !fir.ref<i32>) {
    %fpmap = acc.firstprivate_map varPtr(%private : !fir.ref<i32>) -> !fir.ref<i32>
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: acc.parallel private(%[[PRIVATE]] : !fir.ref<i32>)
    // CHECK-NOT: acc.firstprivate_map
    // CHECK: %[[LOAD:.*]] = fir.load %[[PRIVATE]] : !fir.ref<i32>
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Variable defined outside offload region but firstprivate_map inside -
// should hoist out and then optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_hoist_from_offload_region
func.func @test_hoist_from_offload_region() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_hoistEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel
  acc.parallel {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: fir.call @use_i32(%[[LOAD]])
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Optional variable - should NOT optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_no_hoist
func.func @test_optional_no_hoist(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optionalEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare {{.*}} {{.*}}fortran_attrs<optional>
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map varPtr(%[[DECL]] : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: acc.parallel
  acc.parallel {
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: %[[LOAD:.*]] = fir.load %[[FPMAP]] : !fir.ref<i32>
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Array type (non-trivial) - should NOT optimize

func.func private @use_array(!fir.array<10xi32>)

// CHECK-LABEL: func.func @test_array_no_hoist
func.func @test_array_no_hoist() {
  %c10 = arith.constant 10 : index
  %array = fir.alloca !fir.array<10xi32> {bindc_name = "array_var"}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %decl = fir.declare %array(%shape) uniq_name("_QFtest_arrayEarray_var") : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map varPtr(%[[DECL]] : !fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  // CHECK: acc.parallel
  acc.parallel {
    %load = fir.load %fpmap : !fir.ref<!fir.array<10xi32>>
    // CHECK: %[[LOAD:.*]] = fir.load %[[FPMAP]] : !fir.ref<!fir.array<10xi32>>
    fir.call @use_array(%load) : (!fir.array<10xi32>) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Multiple loads from same firstprivate_map - should optimize and hoist all

func.func private @use_i32_i32(i32, i32)

// CHECK-LABEL: func.func @test_multiple_loads_hoist
func.func @test_multiple_loads_hoist() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_multiple_loadsEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK-DAG: %[[LOAD1:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-DAG: %[[LOAD2:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel
  acc.parallel {
    %load1 = fir.load %fpmap : !fir.ref<i32>
    %load2 = fir.load %fpmap : !fir.ref<i32>
    fir.call @use_i32_i32(%load1, %load2) : (i32, i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Variable through fir.convert - should optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_through_convert
func.func @test_through_convert() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_convertEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %convert = fir.convert %decl : (!fir.ref<i32>) -> !fir.ref<i32>
  %fpmap = acc.firstprivate_map varPtr(%convert : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[CONVERT:.*]] = fir.convert %[[DECL]]
  // CHECK: %[[LOAD:.*]] = fir.load %[[CONVERT]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel
  acc.parallel {
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: fir.call @use_i32(%[[LOAD]])
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Block argument (unknown origin) - should NOT optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_block_arg_no_hoist
func.func @test_block_arg_no_hoist(%arg0: !fir.ref<i32>) {
  // No declare op, so we can't determine if it's optional - conservative no-op
  %fpmap = acc.firstprivate_map varPtr(%arg0 : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map varPtr(%arg0 : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: acc.parallel
  acc.parallel {
    %load = fir.load %fpmap : !fir.ref<i32>
    // CHECK: %[[LOAD:.*]] = fir.load %[[FPMAP]] : !fir.ref<i32>
    fir.call @use_i32(%load) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Optional variable guarded by its presence - should optimize and
// replace the presence check in the region with true

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_hoist
func.func @test_optional_guarded_hoist(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_guardedEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[PRESENT:.*]] = fir.is_present %[[DECL]]
  // CHECK: fir.if %[[PRESENT]] {
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel {
  fir.if %present {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      // CHECK: %[[LOCAL:.*]] = fir.alloca i32
      // CHECK: %[[TRUE:.*]] = arith.constant true
      // CHECK: fir.if %[[TRUE]] {
      // CHECK: fir.store %[[LOAD]] to %[[LOCAL]] : !fir.ref<i32>
      %local = fir.alloca i32
      %inner = fir.is_present %fpmap : (!fir.ref<i32>) -> i1
      fir.if %inner {
        %load = fir.load %fpmap : !fir.ref<i32>
        fir.store %load to %local : !fir.ref<i32>
      }
      %val = fir.load %local : !fir.ref<i32>
      fir.call @use_i32(%val) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable guarded by its presence, with firstprivate_map
// inside the offload region - should hoist out and then optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_hoist_from_offload_region
func.func @test_optional_guarded_hoist_from_offload_region(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_guarded_regionEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: fir.if
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel {
  fir.if %present {
    acc.parallel {
      %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
      %load = fir.load %fpmap : !fir.ref<i32>
      // CHECK: fir.call @use_i32(%[[LOAD]])
      fir.call @use_i32(%load) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable guarded by its presence through an enclosing
// fir.if that is not the immediate parent - should optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_outer_hoist
func.func @test_optional_guarded_outer_hoist(%arg0: !fir.ref<i32>, %flag: i1) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_guarded_outerEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: fir.if
  // CHECK: fir.if
  // CHECK: } else {
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel {
  fir.if %present {
    fir.if %flag {
    } else {
      %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
      acc.parallel {
        %load = fir.load %fpmap : !fir.ref<i32>
        // CHECK: fir.call @use_i32(%[[LOAD]])
        fir.call @use_i32(%load) : (i32) -> ()
        acc.yield
      }
    }
  }
  return
}

// -----

// Test: Optional variable guarded by the presence of the value underlying its
// declare - should optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_declare_input_hoist
func.func @test_optional_guarded_declare_input_hoist(%arg0: !fir.ref<i32>) {
  %present = fir.is_present %arg0 : (!fir.ref<i32>) -> i1
  // CHECK: fir.if
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel {
  fir.if %present {
    %decl = fir.declare %arg0 uniq_name("_QFtest_optional_guarded_declareEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      %load = fir.load %fpmap : !fir.ref<i32>
      // CHECK: fir.call @use_i32(%[[LOAD]])
      fir.call @use_i32(%load) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable in the else region of its presence check - should
// NOT optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_else_no_hoist
func.func @test_optional_else_no_hoist(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_elseEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
  // CHECK: } else {
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map
  // CHECK: acc.parallel {
  // CHECK: %[[INNER:.*]] = fir.is_present %[[FPMAP]]
  // CHECK: fir.if %[[INNER]] {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  fir.if %present {
  } else {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      %local = fir.alloca i32
      %inner = fir.is_present %fpmap : (!fir.ref<i32>) -> i1
      fir.if %inner {
        %load = fir.load %fpmap : !fir.ref<i32>
        fir.store %load to %local : !fir.ref<i32>
      }
      %val = fir.load %local : !fir.ref<i32>
      fir.call @use_i32(%val) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable guarded by the presence of another variable -
// should NOT optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_other_variable_no_hoist
func.func @test_optional_guarded_other_variable_no_hoist(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_otherEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %other = fir.declare %arg1 uniq_name("_QFtest_optional_otherEother_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %other : (!fir.ref<i32>) -> i1
  // CHECK: fir.if
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map
  // CHECK: acc.parallel {
  // CHECK: %[[INNER:.*]] = fir.is_present %[[FPMAP]]
  // CHECK: fir.if %[[INNER]] {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  fir.if %present {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      %local = fir.alloca i32
      %inner = fir.is_present %fpmap : (!fir.ref<i32>) -> i1
      fir.if %inner {
        %load = fir.load %fpmap : !fir.ref<i32>
        fir.store %load to %local : !fir.ref<i32>
      }
      %val = fir.load %local : !fir.ref<i32>
      fir.call @use_i32(%val) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable guarded by a condition other than its presence -
// should NOT optimize

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_not_present_check_no_hoist
func.func @test_optional_guarded_not_present_check_no_hoist(%arg0: !fir.ref<i32>, %flag: i1) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_flagEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: fir.if
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map
  // CHECK: acc.parallel {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  fir.if %flag {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      %load = fir.load %fpmap : !fir.ref<i32>
      fir.call @use_i32(%load) : (i32) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Non-optional variable with a presence check and no guard - should NOT
// optimize, presence checks are only folded under a presence guard

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_presence_check_not_guarded_no_hoist
func.func @test_presence_check_not_guarded_no_hoist() {
  %scalar = fir.alloca i32 {bindc_name = "scalar_var"}
  %decl = fir.declare %scalar uniq_name("_QFtest_presence_checkEscalar_var") : (!fir.ref<i32>) -> !fir.ref<i32>
  %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map
  // CHECK: acc.parallel {
  // CHECK: %[[INNER:.*]] = fir.is_present %[[FPMAP]]
  // CHECK: fir.if %[[INNER]] {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  acc.parallel {
    %local = fir.alloca i32
    %inner = fir.is_present %fpmap : (!fir.ref<i32>) -> i1
    fir.if %inner {
      %load = fir.load %fpmap : !fir.ref<i32>
      fir.store %load to %local : !fir.ref<i32>
    }
    %val = fir.load %local : !fir.ref<i32>
    fir.call @use_i32(%val) : (i32) -> ()
    acc.yield
  }
  return
}

// -----

// Test: Optional variable guarded by its presence, with a use that is neither
// a load nor a presence check - should NOT optimize

func.func private @use_i32(i32)
func.func private @use_ref(!fir.ref<i32>)

// CHECK-LABEL: func.func @test_optional_guarded_other_use_no_hoist
func.func @test_optional_guarded_other_use_no_hoist(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_other_useEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
  // CHECK: fir.if
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map
  // CHECK: acc.parallel {
  // CHECK: %[[INNER:.*]] = fir.is_present %[[FPMAP]]
  // CHECK: fir.if %[[INNER]] {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  // CHECK: fir.call @use_ref(%[[FPMAP]])
  fir.if %present {
    %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
    acc.parallel {
      %local = fir.alloca i32
      %inner = fir.is_present %fpmap : (!fir.ref<i32>) -> i1
      fir.if %inner {
        %load = fir.load %fpmap : !fir.ref<i32>
        fir.store %load to %local : !fir.ref<i32>
      }
      %val = fir.load %local : !fir.ref<i32>
      fir.call @use_i32(%val) : (i32) -> ()
      fir.call @use_ref(%fpmap) : (!fir.ref<i32>) -> ()
      acc.yield
    }
  }
  return
}

// -----

// Test: Optional variable guarded by its presence inside the offload region -
// should NOT optimize, firstprivate_map is hoisted out of the region and thus
// out of the presence guard

func.func private @use_i32(i32)

// CHECK-LABEL: func.func @test_optional_guarded_inside_offload_region_no_hoist
func.func @test_optional_guarded_inside_offload_region_no_hoist(%arg0: !fir.ref<i32>) {
  %decl = fir.declare %arg0 uniq_name("_QFtest_optional_guarded_insideEopt_var") fortran_attrs<optional> : (!fir.ref<i32>) -> !fir.ref<i32>
  // CHECK: %[[DECL:.*]] = fir.declare
  // CHECK: %[[FPMAP:.*]] = acc.firstprivate_map varPtr(%[[DECL]] : !fir.ref<i32>)
  // CHECK: acc.parallel {
  // CHECK: %[[PRESENT:.*]] = fir.is_present %[[DECL]]
  // CHECK: fir.if %[[PRESENT]] {
  // CHECK: fir.load %[[FPMAP]] : !fir.ref<i32>
  acc.parallel {
    %present = fir.is_present %decl : (!fir.ref<i32>) -> i1
    fir.if %present {
      %fpmap = acc.firstprivate_map varPtr(%decl : !fir.ref<i32>) -> !fir.ref<i32>
      %load = fir.load %fpmap : !fir.ref<i32>
      fir.call @use_i32(%load) : (i32) -> ()
    }
    acc.yield
  }
  return
}
