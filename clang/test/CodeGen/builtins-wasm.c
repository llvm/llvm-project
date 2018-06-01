// RUN: %clang_cc1 -triple wasm32-unknown-unknown -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY64

__SIZE_TYPE__ f0(void) {
  return __builtin_wasm_memory_size(0);
// WEBASSEMBLY32: call {{i.*}} @llvm.wasm.memory.size.i32(i32 0)
// WEBASSEMBLY64: call {{i.*}} @llvm.wasm.memory.size.i64(i32 0)
}

__SIZE_TYPE__ f1(__SIZE_TYPE__ delta) {
  return __builtin_wasm_memory_grow(0, delta);
// WEBASSEMBLY32: call i32 @llvm.wasm.memory.grow.i32(i32 0, i32 %{{.*}})
// WEBASSEMBLY64: call i64 @llvm.wasm.memory.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ f2(void) {
  return __builtin_wasm_mem_size(0);
// WEBASSEMBLY32: call {{i.*}} @llvm.wasm.mem.size.i32(i32 0)
// WEBASSEMBLY64: call {{i.*}} @llvm.wasm.mem.size.i64(i32 0)
}

__SIZE_TYPE__ f3(__SIZE_TYPE__ delta) {
  return __builtin_wasm_mem_grow(0, delta);
// WEBASSEMBLY32: call i32 @llvm.wasm.mem.grow.i32(i32 0, i32 %{{.*}})
// WEBASSEMBLY64: call i64 @llvm.wasm.mem.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ f4(void) {
  return __builtin_wasm_current_memory();
// WEBASSEMBLY32: call {{i.*}} @llvm.wasm.current.memory.i32()
// WEBASSEMBLY64: call {{i.*}} @llvm.wasm.current.memory.i64()
}

__SIZE_TYPE__ f5(__SIZE_TYPE__ delta) {
  return __builtin_wasm_grow_memory(delta);
// WEBASSEMBLY32: call i32 @llvm.wasm.grow.memory.i32(i32 %{{.*}})
// WEBASSEMBLY64: call i64 @llvm.wasm.grow.memory.i64(i64 %{{.*}})
}

void f6(unsigned int tag, void *obj) {
  return __builtin_wasm_throw(tag, obj);
// WEBASSEMBLY32: call void @llvm.wasm.throw(i32 %{{.*}}, i8* %{{.*}})
// WEBASSEMBLY64: call void @llvm.wasm.throw(i32 %{{.*}}, i8* %{{.*}})
}

void f7(void) {
  return __builtin_wasm_rethrow();
// WEBASSEMBLY32: call void @llvm.wasm.rethrow()
// WEBASSEMBLY64: call void @llvm.wasm.rethrow()
}
