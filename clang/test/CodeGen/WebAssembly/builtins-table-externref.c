// RUN: %clang_cc1 -triple wasm32 -target-feature +reference-types -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | FileCheck %s -DiPTR=i32
// RUN: %clang_cc1 -triple wasm64 -target-feature +reference-types -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | FileCheck %s -DiPTR=i64
// REQUIRES: webassembly-registered-target

typedef __SIZE_TYPE__ size_t;

static __externref_t table[0];
static const __externref_t const_table[0];

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_get
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call target("wasm.externref") @llvm.wasm.table.get.externref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]])
// CHECK-NEXT:    ret target("wasm.externref") [[TMP0]]
//
__externref_t test_builtin_wasm_table_get(size_t index) {
  return __builtin_wasm_table_get(table, index);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_get_const
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call target("wasm.externref") @llvm.wasm.table.get.externref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]])
// CHECK-NEXT:    ret target("wasm.externref") [[TMP0]]
//
__externref_t test_builtin_wasm_table_get_const(const size_t index) {
  return __builtin_wasm_table_get(table, index);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_set
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.externref") [[REF:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.set.externref.[[iPTR]](ptr addrspace(1) @const_table, [[iPTR]] [[INDEX]], target("wasm.externref") [[REF]])
// CHECK-NEXT:    call void @llvm.wasm.table.set.externref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.externref") [[REF]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_set(const size_t index, __externref_t ref) {
  __builtin_wasm_table_set(const_table, index, ref);
  return __builtin_wasm_table_set(table, index, ref);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_set_const
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.externref") [[REF:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.set.externref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.externref") [[REF]])
// CHECK-NEXT:    call void @llvm.wasm.table.set.externref.[[iPTR]](ptr addrspace(1) @const_table, [[iPTR]] [[INDEX]], target("wasm.externref") [[REF]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_set_const(const size_t index, const __externref_t ref) {
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
// CHECK-SAME: (target("wasm.externref") [[REF:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call [[iPTR]] @llvm.wasm.table.grow.externref.[[iPTR]](ptr addrspace(1) @table, target("wasm.externref") [[REF]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret [[iPTR]] [[TMP0]]
//
size_t test_builtin_wasm_table_grow(__externref_t ref, size_t nelem) {
  return __builtin_wasm_table_grow(table, ref, nelem);
}

// CHECK-LABEL: define {{[^@]+}}@test_builtin_wasm_table_fill
// CHECK-SAME: ([[iPTR]] noundef [[INDEX:%.*]], target("wasm.externref") [[REF:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.fill.externref.[[iPTR]](ptr addrspace(1) @table, [[iPTR]] [[INDEX]], target("wasm.externref") [[REF]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret void
//
void test_builtin_wasm_table_fill(size_t index, __externref_t ref, size_t nelem) {
  __builtin_wasm_table_fill(table, index, ref, nelem);
}

static __externref_t other_table[0];

// CHECK-LABEL: define {{[^@]+}}@test_table_copy
// CHECK-SAME: ([[iPTR]] noundef [[DST_IDX:%.*]], [[iPTR]] noundef [[SRC_IDX:%.*]], [[iPTR]] noundef [[NELEM:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @llvm.wasm.table.copy.[[iPTR]](ptr addrspace(1) @table, ptr addrspace(1) @other_table, [[iPTR]] [[SRC_IDX]], [[iPTR]] [[DST_IDX]], [[iPTR]] [[NELEM]])
// CHECK-NEXT:    ret void
//
void test_table_copy(size_t dst_idx, size_t src_idx, size_t nelem) {
  __builtin_wasm_table_copy(table, other_table, dst_idx, src_idx, nelem);
}
