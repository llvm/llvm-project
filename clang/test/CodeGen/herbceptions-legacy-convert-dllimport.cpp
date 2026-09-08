// Test that the compiler-fabricated legacy-exception-to-herbception
// conversion calls __cxa_error_domain_*_exception_ptr and
// __cxa_error_code_*_exception_ptr with dllimport on Windows targets.

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MINGW
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM

namespace std {
struct error {
  void *d;
  __SIZE_TYPE__ c;
  ~error() noexcept;
};
struct exception_ptr;
}

extern void foo();

// On MSVC, the msvc-prefixed ABI symbols must be dllimport.
// MSVC: declare dllimport ptr @__cxa_error_domain_msvc_exception_ptr()
// MSVC: declare dllimport i64 @__cxa_error_code_msvc_exception_ptr()
// MSVC-NOT: declare ptr @__cxa_error_domain_itanium_exception_ptr()
// MSVC-NOT: declare i64 @__cxa_error_code_itanium_exception_ptr()
void bar() {
  try {
    foo();
  } catch throws(std::error e) {
    (void)e;
  }
}

// On MinGW (Windows Itanium), the itanium-prefixed ABI symbols must be dllimport.
// MINGW: declare dllimport ptr @__cxa_error_domain_itanium_exception_ptr()
// MINGW: declare dllimport i64 @__cxa_error_code_itanium_exception_ptr(
// MINGW-NOT: declare ptr @__cxa_error_domain_msvc_exception_ptr()
// MINGW-NOT: declare i64 @__cxa_error_code_msvc_exception_ptr()
void baz() {
  try {
    foo();
  } catch throws(std::error e) {
    (void)e;
  }
}

// On Linux (non-Windows), no dllimport on the declarations.
// ITANIUM: declare ptr @__cxa_error_domain_itanium_exception_ptr() #{{[0-9]+}}
// ITANIUM: declare i64 @__cxa_error_code_itanium_exception_ptr(ptr noundef) #{{[0-9]+}}
void qux() {
  try {
    foo();
  } catch throws(std::error e) {
    (void)e;
  }
}
