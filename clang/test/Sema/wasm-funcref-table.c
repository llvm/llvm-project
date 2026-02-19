// RUN: %clang_cc1 -triple wasm32 -target-feature +reference-types -fsyntax-only -verify %s

typedef void (*__funcref fn_funcref)(void);

// Valid funcref table declaration (zero-length, static)
static fn_funcref valid_table[0]; // no error expected

// Invalid: non-zero length
static fn_funcref bad_table[1]; // expected-error {{only zero-length WebAssembly tables are currently supported}}

// Array subscript on funcref table should be rejected
void test_subscript(void) {
  (void)valid_table[0]; // expected-error {{cannot subscript a WebAssembly table}}
}

// Original reproducer from https://github.com/llvm/llvm-project/issues/140933
// The declaration should be rejected (not static, non-zero length)
extern fn_funcref issue_table[1]; // expected-error {{WebAssembly table must be static}}
