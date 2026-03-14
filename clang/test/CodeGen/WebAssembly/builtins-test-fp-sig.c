// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +gc -O3 -emit-llvm -DSINGLE_VALUE -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY-SV
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +gc -O3 -emit-llvm -DSINGLE_VALUE -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY-SV
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +gc -target-abi experimental-mv -O3 -emit-llvm  -o - %s 2>&1 | FileCheck %s -check-prefixes WEBASSEMBLY
// RUN: not %clang_cc1 -triple wasm64-unknown-unknown -O3 -emit-llvm -o - %s 2>&1 | FileCheck %s -check-prefixes MISSING-GC

void use(int);

typedef void (*Fvoid)(void);
void test_function_pointer_signature_void(Fvoid func) {
  // MISSING-GC: error: '__builtin_wasm_test_function_pointer_signature' needs target feature gc
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

typedef float (*Ffloats)(float, double, int);
void test_function_pointer_signature_floats(Ffloats func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, float poison, token poison, float poison, double poison, i32 poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

typedef void (*Fpointers)(Fvoid, Ffloats, void*, int*, int***, char[5]);
void test_function_pointer_signature_pointers(Fpointers func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, ptr poison, ptr poison, ptr poison, ptr poison, ptr poison, ptr poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

typedef void (*FVarArgs)(int, ...);
void test_function_pointer_signature_varargs(FVarArgs func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, i32 poison, ptr poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

typedef __externref_t (*FExternRef)(__externref_t, __externref_t);
void test_function_pointer_externref(FExternRef func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, ptr addrspace(10) poison, token poison, ptr addrspace(10) poison, ptr addrspace(10) poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

typedef __funcref Fpointers (*FFuncRef)(__funcref Fvoid, __funcref Ffloats);
void test_function_pointer_funcref(FFuncRef func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, ptr addrspace(20) poison, token poison, ptr addrspace(20) poison, ptr addrspace(20) poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

#ifdef SINGLE_VALUE
// Some tests that we get struct ABIs correct. There is no special code in
// __builtin_wasm_test_function_pointer_signature for this, it gets handled by
// the normal type lowering code.
// Single element structs are unboxed, multi element structs are passed on
// stack.
typedef struct {double x;} (*Fstructs1)(struct {double x;}, struct {float x;}, struct {double x; float y;});
void test_function_pointer_structs1(Fstructs1 func) {
  // WEBASSEMBLY-SV:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, double poison, token poison, double poison, float poison, ptr poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

// Two element return struct ==> return ptr on stack
typedef struct {double x; double y;} (*Fstructs2)(void);
void test_function_pointer_structs2(Fstructs2 func) {
  // WEBASSEMBLY-SV:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, ptr poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

// Return union ==> return ptr on stack, one element union => unboxed
typedef union {double x; float y;} (*FUnions)(union {double x; float y;}, union {double x;});
void test_function_pointer_unions(FUnions func) {
  // WEBASSEMBLY-SV:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, ptr poison, ptr poison, double poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}
#endif
