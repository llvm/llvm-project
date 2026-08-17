// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +multivalue -fsyntax-only -verify=enabled %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature -multivalue -fsyntax-only -verify=disabled %s
// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify=nonwasm %s

// disabled-error@+3 {{'wasm_multivalue' calling convention requires the 'multivalue' target feature to be enabled}}
// nonwasm-warning@+2 {{'wasm_multivalue' calling convention is not supported for this target}}
// nonwasm-warning@+1 2 {{unknown attribute 'wasm_multivalue' ignored}}
__attribute__((wasm_multivalue))
void f1(void);


struct Pair { int a; int b; };

// disabled-error@+3 {{'wasm_multivalue' calling convention requires the 'multivalue' target feature to be enabled}}
// nonwasm-warning@+2 {{'wasm_multivalue' calling convention is not supported for this target}}
// nonwasm-warning@+1 2 {{unknown attribute 'wasm_multivalue' ignored}}
__attribute__((wasm_multivalue))
struct Pair returns_pair(struct Pair x);

// The attribute can be applied to function pointer types.
// disabled-error@+3 {{'wasm_multivalue' calling convention requires the 'multivalue' target feature to be enabled}}
// nonwasm-warning@+2 {{'wasm_multivalue' calling convention is not supported for this target}}
// nonwasm-warning@+1 2 {{unknown attribute 'wasm_multivalue' ignored}}
typedef __attribute__((wasm_multivalue)) struct Pair (*pair_fn_t)(struct Pair);

#if defined(__wasm__)
// Attribute should not take arguments. Only checked on wasm because on other
// targets the attribute is unknown and the diagnostic flow differs.
// enabled-error@+2 {{'wasm_multivalue' attribute takes no arguments}}
// disabled-error@+1 {{'wasm_multivalue' attribute takes no arguments}}
__attribute__((wasm_multivalue(1)))
void takes_no_args(void);
#endif
