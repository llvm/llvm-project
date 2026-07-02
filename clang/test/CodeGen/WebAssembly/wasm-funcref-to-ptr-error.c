// REQUIRES: webassembly-registered-target
// RUN: not %clang_cc1 -triple wasm32 -target-feature +reference-types -S -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple wasm64 -target-feature +reference-types -S -o /dev/null %s 2>&1 | FileCheck %s

// We haven't implemented a way of converting a funcref to a function pointer.
// We can generate code for it if the result is immediately called, which avoids
// the need for creating a function pointer. If the resulting pointer escapes,
// we haven't implemented codegen for that. Diagnose it in the front end rather
// than crashing in the backend.

typedef void (*__funcref funcref_t)(void);
typedef void (*fn_t)(void);

// CHECK: error: a funcref can only be converted to a pointer to be directly called; the resulting pointer cannot otherwise be used
void store_funcref_as_ptr(funcref_t f, fn_t *out) {
  *out = (fn_t)f;
}

// CHECK: error: a funcref can only be converted to a pointer to be directly called; the resulting pointer cannot otherwise be used
fn_t return_funcref_as_ptr(funcref_t f) {
  return (fn_t)f;
}
