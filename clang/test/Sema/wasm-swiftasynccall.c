// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +tail-call -fsyntax-only -verify=tail %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fsyntax-only -verify=notail %s

// tail-no-diagnostics

// swiftasynccall is accepted only with the tail-call feature, which lets the
// backend lower its musttail calls to return_call.

// notail-error@+1 {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) async_func(char *__attribute__((swift_async_context)) ctx) {}
