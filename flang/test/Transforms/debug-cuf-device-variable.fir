// RUN: fir-opt --add-debug-info --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

module {
  func.func @_QPhost(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "a_d", cuf.data_attr = #cuf.cuda<device>}) {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFhostEa"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc1)
    %2 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 {data_attr = #cuf.cuda<device>, uniq_name = "_QFhostEa_d"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc2)
    return
  } loc(#loc3)

  func.func @_QPhostlocals() {
    %0 = fir.alloca i32 {uniq_name = "_QFhostlocalsEd"}
    %1 = fircg.ext_declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFhostlocalsEd"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc4)
    %2 = fir.alloca i32 {uniq_name = "_QFhostlocalsEc"}
    %3 = fircg.ext_declare %2 {data_attr = #cuf.cuda<constant>, uniq_name = "_QFhostlocalsEc"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc5)
    %4 = fir.alloca i32 {uniq_name = "_QFhostlocalsEm"}
    %5 = fircg.ext_declare %4 {data_attr = #cuf.cuda<managed>, uniq_name = "_QFhostlocalsEm"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc6)
    %6 = fir.alloca i32 {uniq_name = "_QFhostlocalsEp"}
    %7 = fircg.ext_declare %6 {data_attr = #cuf.cuda<pinned>, uniq_name = "_QFhostlocalsEp"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc7)
    %8 = fir.alloca i32 {uniq_name = "_QFhostlocalsEu"}
    %9 = fircg.ext_declare %8 {data_attr = #cuf.cuda<unified>, uniq_name = "_QFhostlocalsEu"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc8)
    return
  } loc(#loc9)

  func.func @_QPkernel(%arg0: !fir.ref<i32> {fir.bindc_name = "k_d", cuf.data_attr = #cuf.cuda<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFkernelEk_d"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc10)
    %2 = fir.alloca i32 {uniq_name = "_QFkernelEs"}
    %3 = fircg.ext_declare %2 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFkernelEs"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc11)
    return
  } loc(#loc12)

  func.func @_QPhostdevice(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFhostdeviceEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc13)
    %2 = fir.alloca i32 {uniq_name = "_QFhostdeviceEd"}
    %3 = fircg.ext_declare %2 {data_attr = #cuf.cuda<device>, uniq_name = "_QFhostdeviceEd"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc14)
    return
  } loc(#loc15)

  func.func @_QPsave() {
    %0 = fir.address_of(@_QFsaveEs_d) : !fir.ref<!fir.box<!fir.heap<i32>>>
    %1 = fircg.ext_declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFsaveEs_d"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>> loc(#loc16)
    return
  } loc(#loc17)

  fir.global internal @_QFsaveEs_d {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<i32>> {
    %0 = fir.zero_bits !fir.heap<i32>
    %1 = fir.embox %0 {allocator_idx = 2 : i32} : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
    fir.has_value %1 : !fir.box<!fir.heap<i32>>
  } loc(#loc18)
}

#loc1 = loc("test.cuf":3:1)
#loc2 = loc("test.cuf":4:1)
#loc3 = loc("test.cuf":2:1)
#loc4 = loc("test.cuf":9:1)
#loc5 = loc("test.cuf":10:1)
#loc6 = loc("test.cuf":11:1)
#loc7 = loc("test.cuf":12:1)
#loc8 = loc("test.cuf":13:1)
#loc9 = loc("test.cuf":8:1)
#loc10 = loc("test.cuf":18:1)
#loc11 = loc("test.cuf":19:1)
#loc12 = loc("test.cuf":17:1)
#loc13 = loc("test.cuf":24:1)
#loc14 = loc("test.cuf":25:1)
#loc15 = loc("test.cuf":23:1)
#loc16 = loc("test.cuf":30:1)
#loc17 = loc("test.cuf":29:1)
#loc18 = loc("test.cuf":30:1)

// A variable that gets debug info ends up with a fused loc carrying
// #llvm.di_local_variable, while a filtered one keeps its plain source loc.

// CHECK-LABEL: func.func @_QPhost
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostEa"{{.*}}loc(fused<#llvm.di_local_variable<
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostEa_d"{{.*}}loc("test.cuf":4:1)

// CHECK-LABEL: func.func @_QPhostlocals
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostlocalsEd"{{.*}}loc("test.cuf":9:1)
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostlocalsEc"{{.*}}loc("test.cuf":10:1)
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostlocalsEm"{{.*}}loc(fused<#llvm.di_local_variable<
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostlocalsEp"{{.*}}loc(fused<#llvm.di_local_variable<
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostlocalsEu"{{.*}}loc(fused<#llvm.di_local_variable<

// CHECK-LABEL: func.func @_QPkernel
// CHECK: ext_declare{{.*}}uniq_name = "_QFkernelEk_d"{{.*}}loc(fused<#llvm.di_local_variable<
// CHECK: ext_declare{{.*}}uniq_name = "_QFkernelEs"{{.*}}loc(fused<#llvm.di_local_variable<

// CHECK-LABEL: func.func @_QPhostdevice
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostdeviceEx"{{.*}}loc(fused<#llvm.di_local_variable<
// CHECK: ext_declare{{.*}}uniq_name = "_QFhostdeviceEd"{{.*}}loc("test.cuf":25:1)

// A device variable with SAVE is lowered to a global, leave it unfiltered.

// CHECK-LABEL: func.func @_QPsave
// CHECK: ext_declare{{.*}}uniq_name = "_QFsaveEs_d"{{.*}}loc("test.cuf":30:1)
// CHECK: fir.global internal @_QFsaveEs_d
// CHECK: } loc(fused<[#llvm.di_global_variable_expression<
