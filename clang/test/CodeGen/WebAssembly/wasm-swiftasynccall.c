// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple wasm32-unknown-unknown \
// RUN:   -target-feature +tail-call -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +tail-call \
// RUN:   -S -o - %s | FileCheck %s --check-prefix=ASM

// swiftasynccall uses the swifttailcc CC and musttail calls, which the
// WebAssembly backend lowers to return_call / return_call_indirect with the
// tail-call feature. Sema acceptance: Sema/wasm-swiftasynccall.c.

#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define ASYNC_CONTEXT  __attribute__((swift_async_context))

// Definition uses swifttailcc.
// IR-LABEL: define {{.*}}swifttailcc void @async_leaf(ptr swiftasync
SWIFTASYNCCALL void async_leaf(char *ASYNC_CONTEXT ctx) {
  *ctx += 1;
}

// Direct tail call lowers to return_call.
// IR-LABEL: define {{.*}}swifttailcc void @async_direct(ptr swiftasync
// IR: musttail call swifttailcc void @async_leaf(ptr swiftasync
// IR-NEXT: ret void
//
// ASM-LABEL: async_direct:
// ASM: return_call async_leaf
SWIFTASYNCCALL void async_direct(char *ASYNC_CONTEXT ctx) {
  return async_leaf(ctx);
}

typedef SWIFTASYNCCALL void (*async_fn_t)(char *ASYNC_CONTEXT);

// Indirect tail call lowers to return_call_indirect.
// IR-LABEL: define {{.*}}swifttailcc void @async_indirect(ptr
// IR: musttail call swifttailcc void %{{.*}}(ptr swiftasync
// IR-NEXT: ret void
//
// ASM-LABEL: async_indirect:
// ASM: return_call_indirect
SWIFTASYNCCALL void async_indirect(async_fn_t fn, char *ASYNC_CONTEXT ctx) {
  return fn(ctx);
}
