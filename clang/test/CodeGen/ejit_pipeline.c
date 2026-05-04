// RUN: %clang_cc1 -O2 -emit-llvm -o - %s 2>&1 | FileCheck %s

// Verify full EmbeddedJIT AOT pipeline works via clang.

// PASS1: Bitcode extraction and registration
// CHECK: @__ejit_bitcode = internal constant {{.*}} section ".ejit.bitcode"
// CHECK: @llvm.global_ctors = appending global {{.*}} ptr @ejit_auto_register

// PASS3: Wrapper generation in process_cell
// CHECK: jit_entry:
// CHECK: call ptr @ejit_compile_or_get(
// CHECK: jit_fallback:
// CHECK: jit_dispatch:

// PASS4: Lifecycle handler in lc_handler
// CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @cell_data, i32 0)
// CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cell_data, i32 0)

// PASS1+PASS2: Combined registration in ejit_auto_register
// CHECK-DAG: call void @ejit_register_bitcode(
// CHECK-DAG: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @cell_data, i64 16)

// External runtime declarations
// CHECK-DAG: declare void @ejit_register_bitcode
// CHECK-DAG: declare ptr @ejit_compile_or_get
// CHECK-DAG: declare void @ejit_deactivate_array
// CHECK-DAG: declare void @ejit_activate_array

int cell_data[16] __attribute__((ejit_period_arr("cell")));

__attribute__((ejit_entry))
void process_cell(int __attribute__((ejit_period_arr_ind("cell"))) cell_idx) {
    cell_data[cell_idx]++;
}

__attribute__((ejit_period_lc("cell")))
void lc_handler(int __attribute__((ejit_period_arr_ind("cell"))) cell_idx) {
    cell_data[cell_idx] = 0;
}
