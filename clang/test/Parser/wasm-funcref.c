// RUN: %clang_cc1 -triple powerpc-linux-gnu -fsyntax-only -verify %s

// Test that we trigger an error at parse time if using keyword funcref
// while not using a wasm triple.
typedef void (*__funcref funcref_t)();     // expected-error {{invalid use of '__funcref' keyword outside the WebAssembly triple}}
typedef int (*__funcref fn_funcref_t)(int);// expected-error {{invalid use of '__funcref' keyword outside the WebAssembly triple}}
typedef int (*fn_t)(int);

static fn_funcref_t nullFuncref = 0;
