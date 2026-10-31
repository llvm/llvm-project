// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// Alias analysis classifies delayed-private declarations as allocations even
// though the declaration itself does not allocate storage. Preserve that
// information when assigning TBAA tags.

// A private copy gets allocation tags while the shared/original dummy retains
// dummy-argument tags. Nested copies of the same symbol conservatively share
// the same tag, rather than asserting disjointness between dynamic instances.
// CHECK-DAG: #[[PRIVATE_NODE:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFscalarEx", members
// CHECK-DAG: #[[PRIVATE:.*]] = #llvm.tbaa_tag<base_type = #[[PRIVATE_NODE]], access_type = #[[PRIVATE_NODE]], offset = 0>
// CHECK-DAG: #[[SHARED_NODE:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFscalarEx", members
// CHECK-DAG: #[[SHARED:.*]] = #llvm.tbaa_tag<base_type = #[[SHARED_NODE]], access_type = #[[SHARED_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = private} @scalar : i32

// CHECK-LABEL: func.func @scalar_private(
func.func @scalar_private(%arg: !fir.ref<i32>) {
  %one = arith.constant 1 : i32
  %scope = fir.dummy_scope : !fir.dscope
  %host = fir.declare %arg dummy_scope %scope uniq_name("_QFscalarEx") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  omp.parallel private(@scalar %host -> %p : !fir.ref<i32>) {
    %priv = fir.declare %p uniq_name("_QFscalarEx") : (!fir.ref<i32>) -> !fir.ref<i32>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[PRIVATE]]]}>
    fir.store %one to %priv : !fir.ref<i32>
    // CHECK: fir.load {{.*}} <{tbaa = [#[[SHARED]]]}>
    %v = fir.load %host : !fir.ref<i32>
    omp.parallel private(@scalar %priv -> %p2 : !fir.ref<i32>) {
      %nested = fir.declare %p2 uniq_name("_QFscalarEx") : (!fir.ref<i32>) -> !fir.ref<i32>
      // CHECK: fir.store {{.*}} <{tbaa = [#[[PRIVATE]]]}>
      fir.store %v to %nested : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}
}

// -----

// VALUE copy-in stores must remain in the dummy-argument tree. An OpenMP
// firstprivate instance is a different object, even if VALUE is retained on its
// declaration; it belongs in the allocation tree.
// CHECK-DAG: #[[DUMMY_NODE:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFvalueEx", members
// CHECK-DAG: #[[DUMMY:.*]] = #llvm.tbaa_tag<base_type = #[[DUMMY_NODE]], access_type = #[[DUMMY_NODE]], offset = 0>
// CHECK-DAG: #[[PRIVATE_NODE:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFvalueEx", members
// CHECK-DAG: #[[PRIVATE:.*]] = #llvm.tbaa_tag<base_type = #[[PRIVATE_NODE]], access_type = #[[PRIVATE_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = firstprivate} @first : i32 copy {
^bb0(%mold: !fir.ref<i32>, %copy: !fir.ref<i32>):
  %v = fir.load %mold : !fir.ref<i32>
  fir.store %v to %copy : !fir.ref<i32>
  omp.yield(%copy : !fir.ref<i32>)
}

// CHECK-LABEL: func.func @value_firstprivate(
func.func @value_firstprivate(%arg: i32) {
  %scope = fir.dummy_scope : !fir.dscope
  %storage = fir.alloca i32 {uniq_name = "_QFvalueEx"}
  // CHECK: fir.store {{.*}} <{tbaa = [#[[DUMMY]]]}>
  fir.store %arg to %storage : !fir.ref<i32>
  %host = fir.declare %storage dummy_scope %scope uniq_name("_QFvalueEx") fortran_attrs<value> : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  // CHECK: fir.load {{.*}} <{tbaa = [#[[DUMMY]]]}>
  %v = fir.load %host : !fir.ref<i32>
  omp.parallel private(@first %host -> %p : !fir.ref<i32>) {
    %priv = fir.declare %p uniq_name("_QFvalueEx") fortran_attrs<value> : (!fir.ref<i32>) -> !fir.ref<i32>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[PRIVATE]]]}>
    fir.store %v to %priv : !fir.ref<i32>
    // CHECK: fir.load {{.*}} <{tbaa = [#[[PRIVATE]]]}>
    %loaded = fir.load %priv : !fir.ref<i32>
    omp.terminator
  }
  return
}
}

// -----

// A private automatic array is reached through a descriptor. Tag its data,
// but do not attach data-only tags to descriptor accesses.
// CHECK-DAG: #[[DATA_NODE:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFautomaticEa", members
// CHECK-DAG: #[[DATA:.*]] = #llvm.tbaa_tag<base_type = #[[DATA_NODE]], access_type = #[[DATA_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = private} @array_private : !fir.box<!fir.array<5x?xf64>>
// CHECK-LABEL: func.func @automatic_private(
func.func @automatic_private(%host: !fir.ref<!fir.box<!fir.array<5x?xf64>>>) {
  %one = arith.constant 1 : index
  %v = arith.constant 1.0 : f64
  omp.parallel private(@array_private %host -> %p : !fir.ref<!fir.box<!fir.array<5x?xf64>>>) {
    %priv = fir.declare %p uniq_name("_QFautomaticEa") : (!fir.ref<!fir.box<!fir.array<5x?xf64>>>) -> !fir.ref<!fir.box<!fir.array<5x?xf64>>>
    // CHECK: %[[DESC:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.array<5x?xf64>>>{{$}}
    %desc = fir.load %priv : !fir.ref<!fir.box<!fir.array<5x?xf64>>>
    %element = fir.array_coor %desc %one, %one : (!fir.box<!fir.array<5x?xf64>>, index, index) -> !fir.ref<f64>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[DATA]]]}>
    fir.store %v to %element : !fir.ref<f64>
    // CHECK: fir.load {{.*}} <{tbaa = [#[[DATA]]]}>
    %loaded = fir.load %element : !fir.ref<f64>
    omp.terminator
  }
  return
}
}

// -----

// A private allocatable array is reached through a descriptor. Tag its data,
// but do not attach data-only tags to descriptor accesses.
// CHECK-DAG: #[[DATA_NODE:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFallocatableEa", members
// CHECK-DAG: #[[DATA:.*]] = #llvm.tbaa_tag<base_type = #[[DATA_NODE]], access_type = #[[DATA_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = private} @array_private : !fir.box<!fir.heap<!fir.array<5x?xf64>>>
// CHECK-LABEL: func.func @allocatable_private(
func.func @allocatable_private(%host: !fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>) {
  %one = arith.constant 1 : index
  %v = arith.constant 1.0 : f64
  omp.parallel private(@array_private %host -> %p : !fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>) {
    %priv = fir.declare %p uniq_name("_QFallocatableEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>
    // CHECK: %[[DESC:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>{{$}}
    %desc = fir.load %priv : !fir.ref<!fir.box<!fir.heap<!fir.array<5x?xf64>>>>
    %element = fir.array_coor %desc %one, %one : (!fir.box<!fir.heap<!fir.array<5x?xf64>>>, index, index) -> !fir.ref<f64>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[DATA]]]}>
    fir.store %v to %element : !fir.ref<f64>
    // CHECK: fir.load {{.*}} <{tbaa = [#[[DATA]]]}>
    %loaded = fir.load %element : !fir.ref<f64>
    omp.terminator
  }
  return
}
}

// -----

// FIRSTPRIVATE copies pointer association, not the target data. Its accesses
// must use the target-tree root, which may alias both the original target and
// a private TARGET object. A named allocation tag would be incorrect.
// CHECK-DAG: #[[TARGET_NODE:.*]] = #llvm.tbaa_type_desc<id = "target data", members
// CHECK-DAG: #[[TARGET:.*]] = #llvm.tbaa_tag<base_type = #[[TARGET_NODE]], access_type = #[[TARGET_NODE]], offset = 0>
// CHECK-DAG: #[[NAMED_NODE:.*]] = #llvm.tbaa_type_desc<id = "target data/_QFpointerEt", members = {<#[[TARGET_NODE]], 0>}>
// CHECK-DAG: #[[NAMED:.*]] = #llvm.tbaa_tag<base_type = #[[NAMED_NODE]], access_type = #[[NAMED_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = private} @target_private : f64
omp.private {type = firstprivate} @pointer_recipe : !fir.box<!fir.ptr<f64>> copy {
^bb0(%mold: !fir.ref<!fir.box<!fir.ptr<f64>>>, %copy: !fir.ref<!fir.box<!fir.ptr<f64>>>):
  %desc = fir.load %mold : !fir.ref<!fir.box<!fir.ptr<f64>>>
  fir.store %desc to %copy : !fir.ref<!fir.box<!fir.ptr<f64>>>
  omp.yield(%copy : !fir.ref<!fir.box<!fir.ptr<f64>>>)
}
// CHECK-LABEL: func.func @pointer_firstprivate(
func.func @pointer_firstprivate(%hostp: !fir.ref<!fir.box<!fir.ptr<f64>>>, %hostt: !fir.ref<f64>) {
  %scope = fir.dummy_scope : !fir.dscope
  %t = fir.declare %hostt dummy_scope %scope uniq_name("_QFpointerEt") fortran_attrs<target> : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  // CHECK: fir.load {{.*}} <{tbaa = [#[[TARGET]]]}>
  %v = fir.load %t : !fir.ref<f64>
  omp.parallel private(@pointer_recipe %hostp -> %p, @target_private %t -> %pt : !fir.ref<!fir.box<!fir.ptr<f64>>>, !fir.ref<f64>) {
    %pd = fir.declare %p uniq_name("_QFpointerEp") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<f64>>>) -> !fir.ref<!fir.box<!fir.ptr<f64>>>
    %td = fir.declare %pt uniq_name("_QFpointerEt") fortran_attrs<target> : (!fir.ref<f64>) -> !fir.ref<f64>
    // CHECK: fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f64>>>{{($)}}
    %box = fir.load %pd : !fir.ref<!fir.box<!fir.ptr<f64>>>
    %addr = fir.box_addr %box : (!fir.box<!fir.ptr<f64>>) -> !fir.ptr<f64>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[TARGET]]]}>
    fir.store %v to %addr : !fir.ptr<f64>
    // CHECK: fir.load {{.*}} <{tbaa = [#[[TARGET]]]}>
    %loaded = fir.load %addr : !fir.ptr<f64>
    // CHECK: fir.store {{.*}} <{tbaa = [#[[NAMED]]]}>
    fir.store %loaded to %td : !fir.ref<f64>
    omp.terminator
  }
  return
}
}

// -----

// The declaration can be nested under the clause-carrying operation.
// CHECK-DAG: #[[ARRAY_NODE:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFloopEa", members
// CHECK-DAG: #[[ARRAY:.*]] = #llvm.tbaa_tag<base_type = #[[ARRAY_NODE]], access_type = #[[ARRAY_NODE]], offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>, llvm.data_layout = ""} {
omp.private {type = private} @loop_array : !fir.array<3xf64>
// CHECK-LABEL: func.func @nested_loop_private(
func.func @nested_loop_private(%host: !fir.ref<!fir.array<3xf64>>) {
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : index
  %three = arith.constant 3 : index
  %shape = fir.shape %three : (index) -> !fir.shape<1>
  %v = arith.constant 1.0 : f64
  omp.wsloop private(@loop_array %host -> %p : !fir.ref<!fir.array<3xf64>>) {
    omp.loop_nest (%iv) : i32 = (%one) to (%one) inclusive step (%one) {
      %priv = fir.declare %p(%shape) uniq_name("_QFloopEa") : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      %addr = fir.coordinate_of %priv, %zero : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      // CHECK: fir.store {{.*}} <{tbaa = [#[[ARRAY]]]}>
      fir.store %v to %addr : !fir.ref<f64>
      omp.yield
    }
  }
  return
}
}
