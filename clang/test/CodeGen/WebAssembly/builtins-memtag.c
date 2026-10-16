// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,W32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,W64

typedef __SIZE_TYPE__ size_t;

// --- 0xfc20: memtag.status ---
// CHECK-LABEL: define {{.*}} @test_status()
// W32: call i32 @llvm.wasm.memtag.status.i32(i32 0)
// W64: call i64 @llvm.wasm.memtag.status.i64(i32 0)
size_t test_status() {
  return __builtin_wasm_memtag_status(0);
}

// --- 0xfc21: memtag.extract ---
// CHECK-LABEL: define {{.*}} @test_extract(ptr {{.*}})
// W32: call i32 @llvm.wasm.memtag.extract.i32(i32 0, ptr {{.*}})
// W64: call i64 @llvm.wasm.memtag.extract.i64(i32 0, ptr {{.*}})
size_t test_extract(void *p) {
  return __builtin_wasm_memtag_extract(0, p);
}

// --- 0xfc22: memtag.insert ---
// CHECK-LABEL: define {{.*}} @test_insert(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.insert.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.insert.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *test_insert(void *p, size_t v) {
  return __builtin_wasm_memtag_insert(0, p, v);
}

// --- 0xfc23: memtag.tagbits ---
// CHECK-LABEL: define {{.*}} @memtag_tagbits()
// W32: call i32 @llvm.wasm.memtag.tagbits.i32(i32 0)
// W64: call i64 @llvm.wasm.memtag.tagbits.i64(i32 0)
size_t memtag_tagbits() {
  return __builtin_wasm_memtag_tagbits(0);
}

// --- 0xfc24: memtag.startbit ---
// CHECK-LABEL: define {{.*}} @memtag_startbit()
// W32: call i32 @llvm.wasm.memtag.startbit.i32(i32 0)
// W64: call i64 @llvm.wasm.memtag.startbit.i64(i32 0)
size_t memtag_startbit() {
  return __builtin_wasm_memtag_startbit(0);
}

// --- 0xfc25: memtag.copy ---
// CHECK-LABEL: define {{.*}} @memtag_copy(ptr {{.*}}, ptr {{.*}})
// CHECK: call ptr @llvm.wasm.memtag.copy(i32 0, ptr {{.*}}, ptr {{.*}})
void *memtag_copy(void *d, void *s) {
  return __builtin_wasm_memtag_copy(0, d, s);
}

// --- 0xfc26: memtag.sub ---
// CHECK-LABEL: define {{.*}} @memtag_sub(ptr {{.*}}, ptr {{.*}})
// W32: call i32 @llvm.wasm.memtag.sub.i32(i32 0, ptr {{.*}}, ptr {{.*}})
// W64: call i64 @llvm.wasm.memtag.sub.i64(i32 0, ptr {{.*}}, ptr {{.*}})
size_t memtag_sub(void *a, void *b) {
  return __builtin_wasm_memtag_sub(0, a, b);
}

// --- 0xfc27: memtag.load ---
// CHECK-LABEL: define {{.*}} @memtag_load(ptr {{.*}})
// CHECK: call ptr @llvm.wasm.memtag.load(i32 0, ptr {{.*}})
void *memtag_load(void *p) {
  return __builtin_wasm_memtag_load(0, p);
}

// --- 0xfc28: memtag.untag ---
// CHECK-LABEL: define {{.*}} @memtag_untag(ptr {{.*}})
// CHECK: call ptr @llvm.wasm.memtag.untag(i32 0, ptr {{.*}})
void *memtag_untag(void *p) {
  return __builtin_wasm_memtag_untag(0, p);
}

// --- 0xfc29: memtag.untagstore ---
// CHECK-LABEL: define {{.*}} @memtag_untagstore(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.untagstore.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.untagstore.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *memtag_untagstore(void *p, size_t v) {
  return __builtin_wasm_memtag_untagstore(0, p, v);
}

// --- 0xfc2a: memtag.untagstorez ---
// CHECK-LABEL: define {{.*}} @memtag_untagstorez(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.untagstorez.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.untagstorez.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *memtag_untagstorez(void *p, size_t v) {
  return __builtin_wasm_memtag_untagstorez(0, p, v);
}

// --- 0xfc2b: memtag.store ---
// CHECK-LABEL: define {{.*}} @memtag_store(ptr {{.*}}, {{.*}})
// W32: call void @llvm.wasm.memtag.store.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call void @llvm.wasm.memtag.store.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void memtag_store(void *p, size_t v) {
  __builtin_wasm_memtag_store(0, p, v);
}

// --- 0xfc2c: memtag.storez ---
// CHECK-LABEL: define {{.*}} @memtag_storez(ptr {{.*}}, {{.*}})
// W32: call void @llvm.wasm.memtag.storez.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call void @llvm.wasm.memtag.storez.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void memtag_storez(void *p, size_t v) {
  __builtin_wasm_memtag_storez(0, p, v);
}

// --- 0xfc2d: memtag.random ---
// CHECK-LABEL: define {{.*}} @memtag_random(ptr {{.*}})
// CHECK: call ptr @llvm.wasm.memtag.random(i32 0, ptr {{.*}})
void *memtag_random(void *p) {
  return __builtin_wasm_memtag_random(0, p);
}

// --- 0xfc2e: memtag.randomstore ---
// CHECK-LABEL: define {{.*}} @memtag_randomstore(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.randomstore.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.randomstore.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *memtag_randomstore(void *p, size_t v) {
  return __builtin_wasm_memtag_randomstore(0, p, v);
}

// --- 0xfc2f: memtag.randomstorez ---
// CHECK-LABEL: define {{.*}} @memtag_randomstorez(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.randomstorez.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.randomstorez.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *memtag_randomstorez(void *p, size_t v) {
  return __builtin_wasm_memtag_randomstorez(0, p, v);
}

// --- 0xfc30: memtag.randommask ---
// CHECK-LABEL: define {{.*}} @memtag_randommask(ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.randommask.i32(i32 0, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.randommask.i64(i32 0, ptr {{.*}}, i64 {{.*}})
void *memtag_randommask(void *p, size_t m) {
  return __builtin_wasm_memtag_randommask(0, p, m);
}

// --- 0xfc31: memtag.randommaskstore ---
// CHECK-LABEL: define {{.*}} @memtag_randommaskstore(ptr {{.*}}, {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.randommaskstore.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.randommaskstore.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, i64 {{.*}})
void *memtag_randommaskstore(void *p, size_t v, size_t m) {
  return __builtin_wasm_memtag_randommaskstore(0, p, v, m);
}

// --- 0xfc32: memtag.randommaskstorez ---
// CHECK-LABEL: define {{.*}} @memtag_randommaskstorez(ptr {{.*}}, {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.randommaskstorez.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.randommaskstorez.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, i64 {{.*}})
void *memtag_randommaskstorez(void *p, size_t v, size_t m) {
  return __builtin_wasm_memtag_randommaskstorez(0, p, v, m);
}

// --- 0xfc33: memtag.add ---
// CHECK-LABEL: define {{.*}} @memtag_add(ptr {{.*}}, {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.add.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.add.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, i64 {{.*}})
void *memtag_add(void *p, size_t o, size_t t) {
  return __builtin_wasm_memtag_add(0, p, o, t);
}

// --- 0xfc34: memtag.addstore ---
// CHECK-LABEL: define {{.*}} @memtag_addstore(ptr {{.*}}, {{.*}}, {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.addstore.i32.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.addstore.i64.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}})
void *memtag_addstore(void *p, size_t v, size_t o, size_t t) {
  return __builtin_wasm_memtag_addstore(0, p, v, o, t);
}

// --- 0xfc35: memtag.addstorez ---
// CHECK-LABEL: define {{.*}} @memtag_addstorez(ptr {{.*}}, {{.*}}, {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.addstorez.i32.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.addstorez.i64.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}})
void *memtag_addstorez(void *p, size_t v, size_t o, size_t t) {
  return __builtin_wasm_memtag_addstorez(0, p, v, o, t);
}

// --- 0xfc36: memtag.hint ---
// CHECK-LABEL: define {{.*}} @memtag_hint(ptr {{.*}}, ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.hint.i32(i32 0, ptr {{.*}}, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.hint.i64(i32 0, ptr {{.*}}, ptr {{.*}}, i64 {{.*}})
void *memtag_hint(void *p, void *h, size_t i) {
  return __builtin_wasm_memtag_hint(0, p, h, i);
}

// --- 0xfc37: memtag.hintstore ---
// CHECK-LABEL: define {{.*}} @memtag_hintstore(ptr {{.*}}, {{.*}}, ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.hintstore.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.hintstore.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, ptr {{.*}}, i64 {{.*}})
void *memtag_hintstore(void *p, size_t v, void *h, size_t i) {
  return __builtin_wasm_memtag_hintstore(0, p, v, h, i);
}

// --- 0xfc38: memtag.hintstorez ---
// CHECK-LABEL: define {{.*}} @memtag_hintstorez(ptr {{.*}}, {{.*}}, ptr {{.*}}, {{.*}})
// W32: call ptr @llvm.wasm.memtag.hintstorez.i32.i32(i32 0, ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 {{.*}})
// W64: call ptr @llvm.wasm.memtag.hintstorez.i64.i64(i32 0, ptr {{.*}}, i64 {{.*}}, ptr {{.*}}, i64 {{.*}})
void *memtag_hintstorez(void *p, size_t v, void *h, size_t i) {
  return __builtin_wasm_memtag_hintstorez(0, p, v, h, i);
}
