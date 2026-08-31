// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fsyntax-only -verify=nordc %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify=nordc %s
// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fgpu-rdc -fsyntax-only -verify=rdc %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fgpu-rdc -fsyntax-only -verify=rdc %s

// Tests that the sycl_external attribute is diagnosed as ignored when GPU
// relocatable device code (-fgpu-rdc) is disabled.

// nordc-error@+1{{'clang::sycl_external' attribute is ignored when GPU relocatable device code (-fgpu-rdc) is disabled}}
[[clang::sycl_external]] void foo() {}

// rdc-no-diagnostics
