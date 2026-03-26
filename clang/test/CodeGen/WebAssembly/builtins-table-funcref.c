// RUN: %clang_cc1 -triple wasm32 -target-feature +reference-types -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | FileCheck %s -DiPTR=i32
// RUN: %clang_cc1 -triple wasm64 -target-feature +reference-types -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | FileCheck %s -DiPTR=i64
// REQUIRES: webassembly-registered-target

typedef __SIZE_TYPE__ size_t;

typedef void (*__funcref funcref_t)();
static funcref_t table[0];
static const funcref_t const_table[0];

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_get
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call target("wasm.funcref") @llvm.wasm.table.get.funcref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]])
// CHECK-NEXT:    ret target("wasm.funcref") [[TMP0]]
//
funcref_t test_builtin_wasm_table_get(size_t index) {
  return __builtin_wasm_table_get(table, index);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_get_const
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call target("wasm.funcref") @llvm.wasm.table.get.funcref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]])
// CHECK-NEXT:    ret target("wasm.funcref") [[TMP0]]
//
funcref_t test_builtin_wasm_table_get_const(const size_t index) {
  return __builtin_wasm_table_get(table, index);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_set
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.funcref") noundef [[REF:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.set.funcref.[[iPTR]](ptr addrspace(1) @const_table, [[iPTR]] [[INDEX]], target("wasm.funcref") [[REF]])
// CHECK-NEXT:    call void @llvm.wasm.table.set.funcref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.funcref") [[REF]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_set(const size_t index, funcref_t ref) {
  __builtin_wasm_table_set(const_table, index, ref);
  return __builtin_wasm_table_set(table, index, ref);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_set_const
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.funcref") noundef [[REF:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.set.funcref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.funcref") [[REF]])
// CHECK-NEXT:    call void @llvm.wasm.table.set.funcref.[[iPTR]](ptr addrspace(1) @const_table, [[iPTR]] [[INDEX]], target("wasm.funcref") [[REF]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_set_const(const size_t index, const funcref_t ref) {
  __builtin_wasm_table_set(table, index, ref);
  return __builtin_wasm_table_set(const_table, index, ref);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_size
// CHECK-SAME: () #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call [[iPTR]] @llvm.wasm.table.size.[[iPTR]](ptr addrspace(1) @table)
// CHECK-NEXT:    ret [[iPTR]] [[TMP0]]
//
size_t test_builtin_wasm_table_size() {
  return __builtin_wasm_table_size(table);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_grow
// CHECK-SAME: (target("wasm.funcref") noundef [[REF:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.wasm.table.grow.funcref.[[iPTR]](ptr addrspace(1) @table, target("wasm.funcref") [[REF]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int test_builtin_wasm_table_grow(funcref_t ref, size_t nelem) {
  return __builtin_wasm_table_grow(table, ref, nelem);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_fill
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.funcref") noundef [[REF:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.fill.funcref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.funcref") [[REF]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_fill(size_t index, funcref_t ref, size_t nelem) {
  __builtin_wasm_table_fill(table, index, ref, nelem);
}

static funcref_t other_table[0];

// CHECK-LABEL: define {{[^@]+}}@test_table_copy
// CHECK-SAME: ([[iPTR]] noundef [[DST_IDX:%.*]], [[iPTR]] noundef [[SRC_IDX:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.copy.[[iPTR]](ptr addrspace(1) @table, ptr addrspace(1) @other_table, [[iPTR]] [[SRC_IDX]], [[iPTR]] [[DST_IDX]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret void
//
void test_table_copy(size_t dst_idx, size_t src_idx, size_t nelem) {
  __builtin_wasm_table_copy(table, other_table, dst_idx, src_idx, nelem);
}
