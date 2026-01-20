// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  // Module globals
  fir.global @_QMtestmodEvar_b : i32 {
    %c20_i32 = arith.constant 20 : i32
    fir.has_value %c20_i32 : i32
  }

  fir.global @_QMtestmodEvar_c : i32 {
    %c30_i32 = arith.constant 30 : i32
    fir.has_value %c30_i32 : i32
  }

  fir.global @_QMtestmod2Evar_y : f32 {
    %cst = arith.constant 2.000000e+00 : f32
    fir.has_value %cst : f32
  }

  func.func @_QQmain() attributes {fir.bindc_name = "TEST_USE"} {
    // USE testmod, ONLY: var_b, var_d => var_c
    fir.use_stmt "testmod" only_symbols[[@_QMtestmodEvar_b]] renames[[#fir.use_rename<"var_d", @_QMtestmodEvar_c>]]

    // USE testmod2, var_z => var_y  (no ONLY)
    fir.use_stmt "testmod2" renames[[#fir.use_rename<"var_z", @_QMtestmod2Evar_y>]]

    %0 = fir.address_of(@_QMtestmodEvar_b) : !fir.ref<i32>
    %1 = fircg.ext_declare %0 {uniq_name = "_QMtestmodEvar_b"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_b)

    %2 = fir.address_of(@_QMtestmodEvar_c) : !fir.ref<i32>
    %3 = fircg.ext_declare %2 {uniq_name = "_QMtestmodEvar_c"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_c)

    %4 = fir.address_of(@_QMtestmod2Evar_y) : !fir.ref<f32>
    %5 = fircg.ext_declare %4 {uniq_name = "_QMtestmod2Evar_y"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc_y)

    return
  } loc(#loc_main)
}

#loc_b = loc("test.f90":4:26)
#loc_c = loc("test.f90":4:38)
#loc_y = loc("test.f90":8:24)
#loc_main = loc("test.f90":11:1)

// CHECK-DAG: #[[MOD_TESTMOD:.+]] = #llvm.di_module<{{.*}}name = "testmod"{{.*}}>
// CHECK-DAG: #[[MOD_TESTMOD2:.+]] = #llvm.di_module<{{.*}}name = "testmod2"{{.*}}>

// CHECK-DAG: #[[GVAR_B:.+]] = #llvm.di_global_variable<scope = #[[MOD_TESTMOD]], name = "var_b", linkageName = "_QMtestmodEvar_b"
// CHECK-DAG: #[[GVAR_C:.+]] = #llvm.di_global_variable<scope = #[[MOD_TESTMOD]], name = "var_c", linkageName = "_QMtestmodEvar_c"
// CHECK-DAG: #[[GVAR_Y:.+]] = #llvm.di_global_variable<scope = #[[MOD_TESTMOD2]], name = "var_y", linkageName = "_QMtestmod2Evar_y"

// DISubprogram placeholder (for recursive reference)
// CHECK-DAG: #[[SP_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[RECID:[0-9]+]]]<>, isRecSelf = true{{.*}}name = "TEST_USE"

// 1. Imported declaration without rename (var_b) - has entity but NO name attribute
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP_REC]], entity = #[[GVAR_B]],{{.*}}>

// 2. Imported declaration with rename (var_d => var_c) - has both entity and name
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP_REC]], entity = #[[GVAR_C]],{{.*}}name = "var_d">

// 3. Imported declaration with rename (var_z => var_y) - for module import element
// CHECK-DAG: #[[IMPORT_Z:.+]] = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP_REC]], entity = #[[GVAR_Y]],{{.*}}name = "var_z">

// 4. Imported module (testmod2) with renamed element in its elements field
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_TESTMOD2]]{{.*}}elements = #[[IMPORT_Z]]

// Verify final DISubprogram has retainedNodes (non-empty)
// We don't check the exact order since retainedNodes comes from an unordered collection
// CHECK-DAG: #llvm.di_subprogram<recId = distinct[[[RECID]]]<>{{.*}}name = "TEST_USE"{{.*}}retainedNodes = {{.+}}>
