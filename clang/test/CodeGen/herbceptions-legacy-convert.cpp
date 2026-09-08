// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -o - %s | FileCheck %s --check-prefix=WASM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -exception-model=sjlj -emit-llvm -o - %s | FileCheck %s --check-prefix=SJLJ

// A `try { } catch throws(std::error e)` block also catches legacy C++
// exceptions: calls to noexcept(false) functions inside the try become
// invokes to a catch-all landing pad, which fabricates a std::error
// {__cxa_error_domain_*_exception_ptr(), __cxa_error_code_*_exception_ptr()}
// and routes it to the handler. The compiler emits these extern "C" calls
// directly; no error_domain<std::exception_ptr> specialization or header is
// involved.

namespace std {
struct error {
  void *d;
  __SIZE_TYPE__ c;
  ~error() noexcept;
};
struct exception_ptr;
}

extern void foo();
extern "C" unsigned long long
__cxa_error_code_itanium_exception_ptr(void *);
extern "C" void *__cxa_error_domain_itanium_exception_ptr() noexcept;

// ITANIUM: define dso_local void @_Z3barv() #[[ATTR:[0-9]+]] personality ptr @__gxx_personality_v0 {
// ITANIUM: invoke void @_Z3foov()
// ITANIUM:         to label %invoke.cont unwind label %lpad
// ITANIUM: herb.legacy.convert:
// ITANIUM-NOT: @_ZNSt12error_domainI
// ITANIUM: call ptr @__cxa_error_domain_itanium_exception_ptr()
// ITANIUM: %[[THROWN:[a-z0-9.]+]] = call ptr @__cxa_get_exception_ptr(ptr %exn)
// ITANIUM: call {{.*}}i64 @__cxa_error_code_itanium_exception_ptr(ptr noundef %[[THROWN]])
// ITANIUM: landingpad { ptr, i32 }
// ITANIUM:         catch ptr null
// ITANIUM: br label %herb.legacy.convert
// ITANIUM: catch.throws:
// ITANIUM: call void @_ZNSt5errorD1Ev(ptr {{.*}} %e)
void bar() {
  try {
    foo();
  } catch throws(std::error e) {
    (void)e;
  }
}

// MSVC uses funclet EH: catchswitch/catchpad. The MSVC minting entry point
// reads the current exception itself, so no exception pointer operand is
// passed.
// MSVC: define dso_local void @"?bar@@YAXXZ"() #[[ATTR:[0-9]+]] personality ptr @__CxxFrameHandler3 {
// MSVC: invoke void @"?foo@@YAXXZ"()
// MSVC:         to label %invoke.cont unwind label %catch.dispatch
// MSVC: catchswitch within none [label %herb.legacy.convert]
// MSVC: herb.legacy.convert:
// MSVC: catchpad within
// MSVC-NOT: call ptr @llvm.eh.exceptionpointer
// MSVC: call ptr @__cxa_error_domain_msvc_exception_ptr() [ "funclet"(token
// MSVC: call i64 @__cxa_error_code_msvc_exception_ptr() [ "funclet"(token
// MSVC: br label %catch.throws

// Wasm uses funclet EH with wasm.get.exception storing the object in exn.slot.
// WASM: define void @_Z3barv() {{.*}} personality ptr @__gxx_wasm_personality_v0 {
// WASM: invoke void @_Z3foov()
// WASM:         to label %invoke.cont unwind label %catch.dispatch
// WASM: catchswitch within none [label %catch.start]
// WASM: catchpad within
// WASM: call ptr @llvm.wasm.get.exception(token
// WASM: herb.legacy.convert:
// WASM: call ptr @__cxa_error_domain_itanium_exception_ptr()
// WASM: load ptr, ptr %exn.slot
// WASM: call {{.*}}i32 @__cxa_error_code_itanium_exception_ptr(ptr

// SjLj uses a landingpad plus __cxa_get_exception_ptr.
// SJLJ: define dso_local void @_Z3barv() {{.*}} personality ptr @__gxx_personality_sj0 {
// SJLJ: invoke void @_Z3foov()
// SJLJ: herb.legacy.convert:
// SJLJ: call ptr @__cxa_error_domain_itanium_exception_ptr()
// SJLJ: call ptr @__cxa_get_exception_ptr(ptr %exn)
// SJLJ: call {{.*}}i64 @__cxa_error_code_itanium_exception_ptr(ptr

// ITANIUM: attributes #[[ATTR]] = { {{.*}} }
