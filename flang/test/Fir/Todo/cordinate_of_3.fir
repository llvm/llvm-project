// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// CHECK: not yet implemented fir.array nested inside other array and/or derived type

// `fir.coordinate_of` - `fir.array` inside "boxed" `!fir.type<derived_1{!fir.type<derived_2{}>}` (i.e. nested `!fir.type`)
func @coordinate_box_array_inside_derived(%arg0: !fir.box<!fir.type<derived_1{field_1:!fir.type<derived_2{field_2:!fir.array<10 x i32>}>}>>, %arg1 : index) {
   %idx0 = arith.constant 0 : i32
   %q = fir.coordinate_of %arg0, %idx0, %idx0, %arg1 : (!fir.box<!fir.type<derived_1{field_1:!fir.type<derived_2{field_2:!fir.array<10 x i32>}>}>>, i32, i32, index) -> !fir.ref<f32>
   return
}
