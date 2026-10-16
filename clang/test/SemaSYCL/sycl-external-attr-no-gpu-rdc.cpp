// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fsyntax-only -verify=rdc %s
// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fgpu-rdc -fsyntax-only -verify=rdc %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify=nordc %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fgpu-rdc -fsyntax-only -verify=rdc %s

// Tests that the sycl_external attribute is diagnosed as ignored during device
// compilation without -fgpu-rdc; Host compilation does not diagnose this to
// avoid rediagnosing an issue twice.

// nordc-error@+1{{'clang::sycl_external' attribute is ignored when GPU relocatable device code (-fgpu-rdc) is disabled}}
[[clang::sycl_external]] void foo() {}

// Redeclarations should not result in the diagnostic triggering again:
// nordc-error@+1{{'clang::sycl_external' attribute is ignored when GPU relocatable device code (-fgpu-rdc) is disabled}}
[[clang::sycl_external]] void bar();
void bar();
void bar() {}

// Template instantiations should not result in the diagnostic triggering again:
// nordc-error@+2{{'clang::sycl_external' attribute is ignored when GPU relocatable device code (-fgpu-rdc) is disabled}}
template <typename T>
[[clang::sycl_external]] void ft(T) {}
template void ft(int);
template void ft(long);

// rdc-no-diagnostics
