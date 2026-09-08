// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Big {
  int x[6];
};

void load(struct Big *ptr) {
  // CIR-LABEL: @load
  // LLVM-LABEL: @load

  struct Big b;
  __atomic_load(ptr, &b, __ATOMIC_RELAXED);
  // CIR:      %[[DEST_SLOT:.+]] = cir.alloca "b" align(4) : !cir.ptr<!rec_Big>
  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[PTR_INTPTR:.+]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[DEST_INTPTR:.+]] = cir.cast bitcast %[[DEST_SLOT]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[PTR_VOIDPTR:.+]] = cir.cast bitcast %[[PTR_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[DEST_VOIDPTR:.+]] = cir.cast bitcast %[[DEST_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_load(%[[SIZE]], %[[PTR_VOIDPTR]], %[[DEST_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[DEST:.+]] = alloca %struct.Big
  // LLVM:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @__atomic_load(i64 noundef 24, ptr noundef %[[PTR]], ptr noundef %[[DEST]], i32 noundef 0)
}

void scoped_load(struct Big *ptr) {
  // CIR-LABEL: @scoped_load
  // LLVM-LABEL: @scoped_load

  struct Big b;
  __scoped_atomic_load(ptr, &b, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR:      %[[DEST_SLOT:.+]] = cir.alloca "b" align(4) : !cir.ptr<!rec_Big>
  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[PTR_INTPTR:.+]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[DEST_INTPTR:.+]] = cir.cast bitcast %[[DEST_SLOT]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[PTR_VOIDPTR:.+]] = cir.cast bitcast %[[PTR_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[DEST_VOIDPTR:.+]] = cir.cast bitcast %[[DEST_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_load(%[[SIZE]], %[[PTR_VOIDPTR]], %[[DEST_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[DEST:.+]] = alloca %struct.Big
  // LLVM:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @__atomic_load(i64 noundef 24, ptr noundef %[[PTR]], ptr noundef %[[DEST]], i32 noundef 0)
}

void c11_load(_Atomic(struct Big) *ptr) {
  // CIR-LABEL: @c11_load
  // LLVM-LABEL: @c11_load

  struct Big b = __c11_atomic_load(ptr, __ATOMIC_RELAXED);
  // CIR:      %[[DEST_SLOT:.+]] = cir.alloca "b" align(4) init : !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[TEMP_SLOT:.+]] = cir.alloca "atomic-temp" align(4) : !cir.ptr<!rec_Big>
  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[PTR_INTPTR:.+]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[TEMP_INTPTR:.+]] = cir.cast bitcast %[[TEMP_SLOT]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[PTR_VOIDPTR:.+]] = cir.cast bitcast %[[PTR_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[TEMP_VOIDPTR:.+]] = cir.cast bitcast %[[TEMP_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_load(%[[SIZE]], %[[PTR_VOIDPTR]], %[[TEMP_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()
  // CIR-NEXT: %[[TEMP_CAST:.+]] = cir.cast bitcast %[[TEMP_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!rec_Big>
  // CIR-NEXT: cir.copy %[[TEMP_CAST]] align(4) to %[[DEST_SLOT]] align(4) : !cir.ptr<!rec_Big>

  // LLVM:      %[[DEST_SLOT:.+]] = alloca %struct.Big
  // LLVM-NEXT: %[[TEMP_SLOT:.+]] = alloca %struct.Big
  // LLVM:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @__atomic_load(i64 noundef 24, ptr noundef %[[PTR]], ptr noundef %[[TEMP_SLOT]], i32 noundef 0)
  // LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[DEST_SLOT]], ptr {{.*}}%[[TEMP_SLOT]], i64 24, i1 false)
}

void store(struct Big *dest, struct Big *val) {
  // CIR-LABEL: @store
  // LLVM-LABEL: @store

  __atomic_store(dest, val, __ATOMIC_RELAXED);
  // CIR:      %[[DEST_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[VALUE_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[DEST_INTPTR:.+]] = cir.cast bitcast %[[DEST_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[VALUE_INTPTR:.+]] = cir.cast bitcast %[[VALUE_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[DEST_VOIDPTR:.+]] = cir.cast bitcast %[[DEST_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VALUE_VOIDPTR:.+]] = cir.cast bitcast %[[VALUE_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_store(%[[SIZE]], %[[DEST_VOIDPTR]], %[[VALUE_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[DEST:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[VALUE:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @__atomic_store(i64 noundef 24, ptr noundef %[[DEST]], ptr noundef %[[VALUE]], i32 noundef 0)
}

void scoped_store(struct Big *dest, struct Big *val) {
  // CIR-LABEL: @scoped_store
  // LLVM-LABEL: @scoped_store

  __scoped_atomic_store(dest, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR:      %[[DEST_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[VALUE_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[DEST_INTPTR:.+]] = cir.cast bitcast %[[DEST_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[VALUE_INTPTR:.+]] = cir.cast bitcast %[[VALUE_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[DEST_VOIDPTR:.+]] = cir.cast bitcast %[[DEST_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VALUE_VOIDPTR:.+]] = cir.cast bitcast %[[VALUE_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_store(%[[SIZE]], %[[DEST_VOIDPTR]], %[[VALUE_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[DEST:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[VALUE:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @__atomic_store(i64 noundef 24, ptr noundef %[[DEST]], ptr noundef %[[VALUE]], i32 noundef 0)
}

void c11_store(_Atomic(struct Big) *dest, struct Big *val) {
  // CIR-LABEL: @c11_store
  // LLVM-LABEL: @c11_store

  __c11_atomic_store(dest, *val, __ATOMIC_RELAXED);
  // CIR:      %[[DEST_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[VALUE_PTR:.+]] = cir.load deref align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: cir.copy %[[VALUE_PTR]] align(4) to %[[TEMP_SLOT:.+]] align(4) : !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[DEST_INTPTR:.+]] = cir.cast bitcast %[[DEST_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[VALUE_INTPTR:.+]] = cir.cast bitcast %[[TEMP_SLOT]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.+]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[DEST_VOIDPTR:.+]] = cir.cast bitcast %[[DEST_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VALUE_VOIDPTR:.+]] = cir.cast bitcast %[[VALUE_INTPTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[ORDER:.+]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_store(%[[SIZE]], %[[DEST_VOIDPTR]], %[[VALUE_VOIDPTR]], %[[ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[DEST:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[VALUE:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[TEMP_SLOT:.+]], ptr {{.*}}%[[VALUE]], i64 24, i1 false)
  // LLVM-NEXT: call void @__atomic_store(i64 noundef 24, ptr noundef %[[DEST]], ptr noundef %[[TEMP_SLOT]], i32 noundef 0)
}

void exchange(struct Big *ptr, struct Big *val, struct Big *old) {
  // CIR-LABEL: @exchange
  // LLVM-LABEL: @exchange

  __atomic_exchange(ptr, val, old, __ATOMIC_RELAXED);
  // CIR-NEXT: %[[PTR:.*]] = cir.alloca "ptr" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR-NEXT: %[[VAL:.*]] = cir.alloca "val" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR-NEXT: %[[OLD:.*]] = cir.alloca "old" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR:      %[[LOAD_PTR:.*]] = cir.load align(8) %[[PTR]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[LOAD_VAL:.*]] = cir.load align(8) %[[VAL]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[LOAD_OLD:.*]] = cir.load align(8) %[[OLD]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[CAST_PTR:.*]] = cir.cast bitcast %[[LOAD_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_VAL:.*]] = cir.cast bitcast %[[LOAD_VAL]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_OLD:.*]] = cir.cast bitcast %[[LOAD_OLD]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.*]] = cir.const #cir.int<24> : !u64i 
  // CIR-NEXT: %[[VOID_PTR:.*]] = cir.cast bitcast %[[CAST_PTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_VAL:.*]] = cir.cast bitcast %[[CAST_VAL]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_OLD:.*]] = cir.cast bitcast %[[CAST_OLD]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[MEM_ORDER:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_exchange(%[[SIZE]], %[[VOID_PTR]], %[[VOID_VAL]], %[[VOID_OLD]], %[[MEM_ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[PTR:.*]] = alloca ptr, align 8
  // LLVM-NEXT: %[[VAL:.*]] = alloca ptr, align 8
  // LLVM-NEXT: %[[OLD:.*]] = alloca ptr, align 8
  // LLVM:      %[[LOAD_PTR:.*]] = load ptr, ptr %[[PTR]], align 8
  // LLVM-NEXT: %[[LOAD_VAL:.*]] = load ptr, ptr %[[VAL]], align 8
  // LLVM-NEXT: %[[LOAD_OLD:.*]] = load ptr, ptr %[[OLD]], align 8
  // LLVM-NEXT: call void @__atomic_exchange(i64 noundef 24, ptr noundef %[[LOAD_PTR]], ptr noundef %[[LOAD_VAL]], ptr noundef %[[LOAD_OLD]], i32 noundef 0)
}

void scoped_exchange(struct Big *ptr, struct Big *val, struct Big *old) {
  // CIR-LABEL: @scoped_exchange
  // LLVM-LABEL: @scoped_exchange

  __scoped_atomic_exchange(ptr, val, old, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-NEXT: %[[PTR:.*]] = cir.alloca "ptr" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR-NEXT: %[[VAL:.*]] = cir.alloca "val" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR-NEXT: %[[OLD:.*]] = cir.alloca "old" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR:      %[[LOAD_PTR:.*]] = cir.load align(8) %[[PTR]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[LOAD_VAL:.*]] = cir.load align(8) %[[VAL]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[LOAD_OLD:.*]] = cir.load align(8) %[[OLD]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[CAST_PTR:.*]] = cir.cast bitcast %[[LOAD_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_VAL:.*]] = cir.cast bitcast %[[LOAD_VAL]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_OLD:.*]] = cir.cast bitcast %[[LOAD_OLD]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.*]] = cir.const #cir.int<24> : !u64i 
  // CIR-NEXT: %[[VOID_PTR:.*]] = cir.cast bitcast %[[CAST_PTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_VAL:.*]] = cir.cast bitcast %[[CAST_VAL]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_OLD:.*]] = cir.cast bitcast %[[CAST_OLD]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[MEM_ORDER:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_exchange(%[[SIZE]], %[[VOID_PTR]], %[[VOID_VAL]], %[[VOID_OLD]], %[[MEM_ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[PTR:.*]] = alloca ptr, align 8
  // LLVM-NEXT: %[[VAL:.*]] = alloca ptr, align 8
  // LLVM-NEXT: %[[OLD:.*]] = alloca ptr, align 8
  // LLVM:      %[[LOAD_PTR:.*]] = load ptr, ptr %[[PTR]], align 8
  // LLVM-NEXT: %[[LOAD_VAL:.*]] = load ptr, ptr %[[VAL]], align 8
  // LLVM-NEXT: %[[LOAD_OLD:.*]] = load ptr, ptr %[[OLD]], align 8
  // LLVM-NEXT: call void @__atomic_exchange(i64 noundef 24, ptr noundef %[[LOAD_PTR]], ptr noundef %[[LOAD_VAL]], ptr noundef %[[LOAD_OLD]], i32 noundef 0)
}

struct Big c11_exchange(_Atomic(struct Big) *ptr, struct Big val) {
  // CIR-LABEL: @c11_exchange
  // LLVM-LABEL: @c11_exchange

  return __c11_atomic_exchange(ptr, val, __ATOMIC_RELAXED);
  // CIR:      %[[PTR:.*]] = cir.alloca "ptr" align(8) init : !cir.ptr<!cir.ptr<!rec_Big>>
  // CIR-NEXT: %[[VAL:.*]] = cir.alloca "val" align(4) init : !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[TMP1:.*]] = cir.alloca ".atomictmp" align(4) : !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[TMP2:.*]] = cir.alloca "atomic-temp" align(4) : !cir.ptr<!rec_Big>
  // CIR:      %[[LOAD_PTR:.*]] = cir.load align(8) %[[PTR]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
  // CIR-NEXT: cir.copy %[[VAL]] align(4) to %[[TMP1]] align(4) : !cir.ptr<!rec_Big>
  // CIR-NEXT: %[[CAST_PTR:.*]] = cir.cast bitcast %[[LOAD_PTR]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_TMP1:.*]] = cir.cast bitcast %[[TMP1]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[CAST_TMP2:.*]] = cir.cast bitcast %[[TMP2]] : !cir.ptr<!rec_Big> -> !cir.ptr<!cir.int<u, 192>>
  // CIR-NEXT: %[[SIZE:.*]] = cir.const #cir.int<24> : !u64i
  // CIR-NEXT: %[[VOID_PTR:.*]] = cir.cast bitcast %[[CAST_PTR]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_TMP1:.*]] = cir.cast bitcast %[[CAST_TMP1]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[VOID_TMP2:.*]] = cir.cast bitcast %[[CAST_TMP2]] : !cir.ptr<!cir.int<u, 192>> -> !cir.ptr<!void>
  // CIR-NEXT: %[[MEM_ORDER:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.call @__atomic_exchange(%[[SIZE]], %[[VOID_PTR]], %[[VOID_TMP1]], %[[VOID_TMP2]], %[[MEM_ORDER]]) : (!u64i {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}) -> ()

  // LLVM:      %[[LOAD_PTR:.*]] = load ptr, ptr %{{.*}}, align 8
  // LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 24, i1 false)
  // LLVM-NEXT: call void @__atomic_exchange(i64 noundef 24, ptr noundef %[[LOAD_PTR]], ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 0)
  // LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 24, i1 false)
}
