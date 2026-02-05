// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda- -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spir64 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64 -fsyntax-only -verify %s

[[clang::device_kernel]] void kernel1() {}

namespace {
[[clang::device_kernel]] void kernel2() {} // expected-error {{'kernel2' is specified as a device kernel but it is not externally visible}}
}

namespace ns {
  [[clang::device_kernel]] void kernel3() {}
}

[[clang::device_kernel]] static void kernel4() {} // expected-error {{'kernel4' is specified as a device kernel but it is not externally visible}}
