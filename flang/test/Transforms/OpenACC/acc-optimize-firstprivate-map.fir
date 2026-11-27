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

// Test: Optional variable used as a loop bound in a compute construct guarded
// by its presence, after recipe materialization - should optimize and replace
// the presence checks in the region with true

// CHECK-LABEL: func.func @test_optional_guarded_hoist
func.func @test_optional_guarded_hoist(%n_arg: !fir.ref<i32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %dscope = fir.dummy_scope : !fir.dscope
  %n = fir.declare %n_arg dummy_scope %dscope arg 1 uniq_name("_QFtest_optional_guardedEn") fortran_attrs<intent_in, optional> : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %n_is_present = fir.is_present %n : (!fir.ref<i32>) -> i1
  // CHECK: %[[N:.*]] = fir.declare
  // CHECK: %[[N_IS_PRESENT:.*]] = fir.is_present %[[N]]
  // CHECK: fir.if %[[N_IS_PRESENT]] {
  // CHECK: %[[N_VALUE:.*]] = fir.load %[[N]] : !fir.ref<i32>
  // CHECK-NOT: acc.firstprivate_map
  // CHECK: acc.parallel combined(loop) {
  fir.if %n_is_present {
    %n_mapped = acc.firstprivate_map varPtr(%n : !fir.ref<i32>) implicit(true) name("n") -> !fir.ref<i32>
    acc.parallel combined(loop) {
      // CHECK: %[[N_PRIVATE:.*]] = fir.alloca i32
      // CHECK-NOT: fir.is_present
      // CHECK: %[[SELECT_TRUE:.*]] = arith.constant true
      // CHECK: %[[N_TARGET:.*]] = fir.if %[[SELECT_TRUE]] -> (!fir.ref<i32>) {
      // CHECK: fir.result %[[N_PRIVATE]] : !fir.ref<i32>
      // CHECK-NOT: fir.is_present
      // CHECK: %[[STORE_TRUE:.*]] = arith.constant true
      // CHECK: fir.if %[[STORE_TRUE]] {
      // CHECK: fir.store %[[N_VALUE]] to %[[N_TARGET]] : !fir.ref<i32>
      // CHECK: %[[N_PRIVATE_VALUE:.*]] = fir.load %[[N_TARGET]] : !fir.ref<i32>
      // CHECK: %[[UPPER:.*]] = arith.subi %[[N_PRIVATE_VALUE]]
      // CHECK: acc.loop {{.*}} to (%[[UPPER]] : i32)
      %n_private = fir.alloca i32 {acc.var_name = #acc.var_name<"n">}
      %select_present = fir.is_present %n_mapped : (!fir.ref<i32>) -> i1
      %n_target = fir.if %select_present -> (!fir.ref<i32>) {
        fir.result %n_private : !fir.ref<i32>
      } else {
        %n_absent = fir.absent !fir.ref<i32>
        fir.result %n_absent : !fir.ref<i32>
      } {acc.var_name = #acc.var_name<"n">}
      %store_present = fir.is_present %n_mapped : (!fir.ref<i32>) -> i1
      fir.if %store_present {
        %n_mapped_value = fir.load %n_mapped : !fir.ref<i32>
        fir.store %n_mapped_value to %n_target : !fir.ref<i32>
      }
      %n_private_value = fir.load %n_target : !fir.ref<i32>
      %upper = arith.subi %n_private_value, %c1_i32 : i32
      acc.loop combined(parallel) control(%i : i32) = (%c0_i32 : i32) to (%upper : i32) step (%c1_i32 : i32) {
        acc.yield
      } inclusiveUpperbound(array<i1: true>) independent
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

// -----

// A descriptor used only by fir.box_dims should have its firstprivate_map
// operation removed and be rebuilt with a null base using dimensions read before
// the offload region.

func.func private @use_dims(index, index, index)

// CHECK-LABEL: func.func @descriptor_without_base
// CHECK-NOT: acc.firstprivate_map
// CHECK: %[[HOSTBOX:.*]] = fir.load
// CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[HOSTBOX]]
// CHECK-NEXT: acc.parallel {
// CHECK-NEXT: %[[NULL:.*]] = fir.zero_bits
// CHECK-NEXT: %[[BOX:.*]] = fir.create_box %[[NULL]] lbs(%[[DIMS]]#0) extents(%[[DIMS]]#1) strides(%[[DIMS]]#2)
// CHECK-NEXT: %[[DEVICE_DIMS:.*]]:3 = fir.box_dims %[[BOX]]
// CHECK-NEXT: fir.call @use_dims(%[[DEVICE_DIMS]]#0, %[[DEVICE_DIMS]]#1, %[[DEVICE_DIMS]]#2)
// CHECK-NEXT: acc.yield
func.func @descriptor_without_base(%init: !fir.box<!fir.array<?xf64>>) {
  %c0 = arith.constant 0 : index
  %host = fir.alloca !fir.box<!fir.array<?xf64>>
  fir.store %init to %host : !fir.ref<!fir.box<!fir.array<?xf64>>>
  %map = acc.firstprivate_map varPtr(%host : !fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
  acc.parallel {
    %box = fir.load %map : !fir.ref<!fir.box<!fir.array<?xf64>>>
    %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
    fir.call @use_dims(%dims#0, %dims#1, %dims#2) : (index, index, index) -> ()
    acc.yield
  }
  return
}

// -----

// A descriptor used by fir.box_dims and fir.box_addr should retain its
// firstprivate_map operation and be rebuilt from host dimensions and the mapped
// descriptor's device base.

func.func private @use_array(!fir.ref<!fir.array<?xf64>>, index, index, index)

// CHECK-LABEL: func.func @descriptor_with_base
// CHECK: %[[MAP:.*]] = acc.firstprivate_map
// CHECK: %[[HOSTBOX:.*]] = fir.load
// CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[HOSTBOX]]
// CHECK-NEXT: acc.parallel {
// CHECK-NEXT: %[[DEVICEBOX:.*]] = fir.load %[[MAP]]
// CHECK-NEXT: %[[BASE:.*]] = fir.box_addr %[[DEVICEBOX]]
// CHECK-NEXT: %[[BOX:.*]] = fir.create_box %[[BASE]] lbs(%[[DIMS]]#0) extents(%[[DIMS]]#1) strides(%[[DIMS]]#2)
// CHECK-NEXT: %[[DEVICE_DIMS:.*]]:3 = fir.box_dims %[[BOX]]
// CHECK-NEXT: %[[DEVICE_BASE:.*]] = fir.box_addr %[[BOX]]
// CHECK-NEXT: fir.call @use_array(%[[DEVICE_BASE]], %[[DEVICE_DIMS]]#0, %[[DEVICE_DIMS]]#1, %[[DEVICE_DIMS]]#2)
// CHECK-NEXT: acc.yield
func.func @descriptor_with_base(%init: !fir.box<!fir.array<?xf64>>) {
  %c0 = arith.constant 0 : index
  %host = fir.alloca !fir.box<!fir.array<?xf64>>
  fir.store %init to %host : !fir.ref<!fir.box<!fir.array<?xf64>>>
  %map = acc.firstprivate_map varPtr(%host : !fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
  acc.parallel {
    %box = fir.load %map : !fir.ref<!fir.box<!fir.array<?xf64>>>
    %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
    %base = fir.box_addr %box : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
    fir.call @use_array(%base, %dims#0, %dims#1, %dims#2) : (!fir.ref<!fir.array<?xf64>>, index, index, index) -> ()
    acc.yield
  }
  return
}

// -----

// An OPTIONAL descriptor without a presence check should retain its
// firstprivate_map operation and descriptor load without scalarization.

func.func private @use_extent(index)

// CHECK-LABEL: func.func @optional_descriptor_unguarded
// CHECK: %[[HOST:.*]] = fir.declare
// CHECK-NEXT: %[[MAP:.*]] = acc.firstprivate_map varPtr(%[[HOST]] :
// CHECK-NEXT: acc.parallel {
// CHECK-NEXT: %[[BOX:.*]] = fir.load %[[MAP]]
// CHECK-NEXT: %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]]
// CHECK-NEXT: fir.call @use_extent(%[[DIMS]]#1)
func.func @optional_descriptor_unguarded(%arg0: !fir.ref<!fir.box<!fir.array<?xf64>>>) {
  %c0 = arith.constant 0 : index
  %host = fir.declare %arg0 uniq_name("_QFoptional_descriptor_unguardedEa") fortran_attrs<optional> : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
  %map = acc.firstprivate_map varPtr(%host : !fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
  acc.parallel {
    %box = fir.load %map : !fir.ref<!fir.box<!fir.array<?xf64>>>
    %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
    fir.call @use_extent(%dims#1) : (index) -> ()
    acc.yield
  }
  return
}

// -----

// An OPTIONAL descriptor whose firstprivate_map operation is presence-guarded
// should be scalarized

func.func private @use_extent(index)

// CHECK-LABEL: func.func @optional_descriptor_guarded
// CHECK: %[[HOST:.*]] = fir.declare
// CHECK-NEXT: %[[PRESENT:.*]] = fir.is_present %[[HOST]]
// CHECK-NEXT: fir.if %[[PRESENT]] {
// CHECK-NEXT: %[[HOSTBOX:.*]] = fir.load %[[HOST]]
// CHECK-NOT: acc.firstprivate_map
// CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[HOSTBOX]]
// CHECK-NEXT: acc.parallel {
// CHECK-NEXT: %[[NULL:.*]] = fir.zero_bits
// CHECK-NEXT: %[[BOX:.*]] = fir.create_box %[[NULL]] lbs(%[[DIMS]]#0) extents(%[[DIMS]]#1) strides(%[[DIMS]]#2)
// CHECK-NEXT: %[[DEVICE_DIMS:.*]]:3 = fir.box_dims %[[BOX]]
// CHECK-NEXT: fir.call @use_extent(%[[DEVICE_DIMS]]#1)
func.func @optional_descriptor_guarded(%arg0: !fir.ref<!fir.box<!fir.array<?xf64>>>) {
  %c0 = arith.constant 0 : index
  %host = fir.declare %arg0 uniq_name("_QFoptional_descriptor_guardedEa") fortran_attrs<optional> : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
  %present = fir.is_present %host : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> i1
  fir.if %present {
    %map = acc.firstprivate_map varPtr(%host : !fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.array<?xf64>>>
    acc.parallel {
      %box = fir.load %map : !fir.ref<!fir.box<!fir.array<?xf64>>>
      %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
      fir.call @use_extent(%dims#1) : (index) -> ()
      acc.yield
    }
  }
  return
}
