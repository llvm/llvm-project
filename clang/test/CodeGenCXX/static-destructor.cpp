// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -o - | FileCheck --check-prefix=X86 %s
// RUN: %clang_cc1 %s -triple=wasm32 -emit-llvm -o - | FileCheck --check-prefix=WASM %s
// RUN: %clang_cc1 %s -triple=armv7-apple-darwin9 -emit-llvm -o - | FileCheck --check-prefix=ARM %s
// RUN: %clang_cc1 %s -triple=wasm32 -emit-llvm -fno-use-cxa-atexit -DTLS -o - | FileCheck --check-prefix=WASM-TLS %s

// Test that destructors are not passed directly to __cxa_atexit when their
// signatures do not match the type of its first argument.
// e.g. ARM and WebAssembly have destructors that return this instead of void.


class Foo {
 public:
  ~Foo() {
  }
};

Foo global;

// X86 destructors have void return, and are registered directly with __cxa_atexit.
// X86: define internal void @__cxx_global_var_init()
// X86:   call i32 @__cxa_atexit(ptr @_ZN3FooD1Ev, ptr @global, ptr @__dso_handle)

// ARM destructors return this, but can be registered directly with __cxa_atexit
// because the calling conventions tolerate the mismatch.
// ARM: define internal void @__cxx_global_var_init()
// ARM:   call i32 @__cxa_atexit(ptr @_ZN3FooD1Ev, ptr @global, ptr @__dso_handle)

// Wasm destructors return this, and use a wrapper function, which is registered
// with __cxa_atexit.
// WASM: define internal void @__cxx_global_var_init()
// WASM: call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)

// WASM: define internal void @__cxx_global_array_dtor(ptr noundef %0)
// WASM: %call = call noundef ptr @_ZN3FooD1Ev(ptr {{[^,]*}} @global)

// The same holds for temporaries that are lifetime-extended by a reference with
// static storage duration.
const Foo &global_ref = Foo();

// X86: define internal void @__cxx_global_var_init.1()
// X86:   call i32 @__cxa_atexit(ptr @_ZN3FooD1Ev, ptr @_ZGR10global_ref_, ptr @__dso_handle)

// ARM: define internal void @__cxx_global_var_init.1()
// ARM:   call i32 @__cxa_atexit(ptr @_ZN3FooD1Ev, ptr @_ZGR10global_ref_, ptr @__dso_handle)

// WASM: define internal void @__cxx_global_var_init.1()
// WASM: call i32 @__cxa_atexit(ptr @[[REF_DTOR:__cxx_global_array_dtor[.0-9]*]], ptr null, ptr @__dso_handle)

// WASM: define internal void @[[REF_DTOR]](ptr noundef %0)
// WASM: %call = call noundef ptr @_ZN3FooD1Ev(ptr {{[^,]*}} @_ZGR10global_ref_)

// Thread-local ones are registered with `__cxa_thread_atexit`, even with
// `-fno-use-cxa-atexit`.
#ifdef TLS
thread_local const Foo &tls_ref = Foo();
#endif

// WASM-TLS: call i32 @__cxa_thread_atexit(ptr @[[TLS_DTOR:__cxx_global_array_dtor[.0-9]*]], ptr null, ptr @__dso_handle)

// WASM-TLS: define internal void @[[TLS_DTOR]](ptr noundef %0)
// WASM-TLS: %call = call noundef ptr @_ZN3FooD1Ev(ptr {{[^,]*}} @_ZGR7tls_ref_)
