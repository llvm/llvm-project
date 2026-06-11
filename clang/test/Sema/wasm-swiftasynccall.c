// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +tail-call -fsyntax-only -verify=tail %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fsyntax-only -verify=notail %s

// tail-no-diagnostics

// With the tail-call feature, swiftasynccall should be accepted on
// WebAssembly since the backend can lower musttail calls to return_call.
// Without it, swiftasynccall should be rejected.

// notail-error@+1 {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) async_func(char *__attribute__((swift_async_context)) ctx) {}
