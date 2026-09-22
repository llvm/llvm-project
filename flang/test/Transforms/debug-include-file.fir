// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Lowering fuses the inclusion information onto the location of anything read
// through an INCLUDE statement, which is the shape reproduced below. Check that
// a procedure, its dummy argument and a global carrying such a location still
// get debug information, and that it names the file and line where they are
// written rather than the INCLUDE statement.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPincluded_sub(%arg0: !fir.ref<i32>) {
    %0 = fir.undefined !fir.dscope loc(#sub_in_body)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFincluded_subEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#arg_in_body)
    return loc(#ret_in_body)
  } loc(#sub_in_body)

  fir.global @_QMmEmodvar : i32 {
    %c7_i32 = arith.constant 7 : i32 loc(#global_in_body)
    fir.has_value %c7_i32 : i32 loc(#global_in_body)
  } loc(#global_in_body)

  func.func @_QQmain() {
    return loc(#ret_in_main)
  } loc(#prog_in_main)
} loc(#module_loc)

#module_loc = loc("main.f90":0:0)
// The INCLUDE statement itself, on line 1 of the including file.
#include_stmt = loc("main.f90":1:1)

// Written in the included file, reached through the INCLUDE statement.
#sub_pos = loc("body.f90":4:1)
#sub_in_body = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#sub_pos, #include_stmt])
#arg_pos = loc("body.f90":5:14)
#arg_in_body = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#arg_pos, #include_stmt])
#ret_pos = loc("body.f90":6:1)
#ret_in_body = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#ret_pos, #include_stmt])
#global_pos = loc("body.f90":2:14)
#global_in_body = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#global_pos, #include_stmt])

// Written in the including file, for contrast.
#prog_in_main = loc("main.f90":2:1)
#ret_in_main = loc("main.f90":4:1)

// CHECK-DAG: #[[MAIN:.*]] = #llvm.di_file<"main.f90" in "">
// CHECK-DAG: #[[BODY:.*]] = #llvm.di_file<"body.f90" in "">
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "included_sub", linkageName = "_QPincluded_sub", file = #[[BODY]], line = 4, scopeLine = 4{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "_QQmain", linkageName = "_QQmain", file = #[[MAIN]], line = 2, scopeLine = 2{{.*}}>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "i", file = #[[BODY]], line = 5{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "modvar", linkageName = "_QMmEmodvar", file = #[[BODY]], line = 2{{.*}}>
