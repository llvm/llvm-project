// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -std=c++20 -fsyntax-only -fno-spell-checking -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -std=c++20 -fsyntax-only -fno-spell-checking -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgpu9.00-amd-amdhsa -x hip -std=c++20 -fsyntax-only -fno-spell-checking -verify %s
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -x hip -std=c++20 -fsyntax-only -fno-spell-checking -verify %s

#include "Inputs/cuda.h"

namespace associated {
struct Arg {};
__global__ void kernel(Arg);
template <class T> __global__ void template_kernel(T);
void host_function(Arg);
} // namespace associated

void test(associated::Arg arg) {
  // Parentheses and qualification still suppress ADL.
  (kernel)<<<1, 1>>>(arg); // expected-error {{use of undeclared identifier 'kernel'}}
  ::kernel<<<1, 1>>>(arg); // expected-error {{no member named 'kernel' in the global namespace}}
  // The address-of operand is the whole call, not the kernel name.
  &kernel<<<1, 1>>>(arg); // expected-error {{cannot take the address of an rvalue of type 'void'}}
  &template_kernel<associated::Arg><<<1, 1>>>(arg); // expected-error {{cannot take the address of an rvalue of type 'void'}}
  // Finding a function by ADL does not make it a kernel.
  host_function<<<1, 1>>>(arg); // expected-error {{kernel call to non-global function 'host_function'}}
}

void block_scope(associated::Arg arg) {
  __global__ void kernel(int); // expected-note {{passing argument to parameter here}}
  kernel<<<1, 1>>>(arg); // expected-error {{no viable conversion from 'associated::Arg' to 'int'}}
}

struct Member {
  static __global__ void kernel(int); // expected-note {{passing argument to parameter here}}
  void test(associated::Arg arg) {
    kernel<<<1, 1>>>(arg); // expected-error {{no viable conversion from 'associated::Arg' to 'int'}}
  }
};

namespace config {
struct Grid {
  operator dim3() const;
};
__global__ void only_in_config();
} // namespace config

void configuration() {
  // Only the kernel arguments, not the execution configuration, supply
  // associated namespaces for lookup of the kernel.
  only_in_config<<<config::Grid{}, 1>>>(); // expected-error {{use of undeclared identifier 'only_in_config'}}
}

namespace associated {
__global__ void ambiguous(Arg); // expected-note {{candidate function}}
} // namespace associated

__global__ void ambiguous(associated::Arg); // expected-note {{candidate function}}

void overloads(associated::Arg arg) {
  // ADL augments ordinary lookup rather than replacing its candidates.
  ambiguous<<<1, 1>>>(arg); // expected-error {{call to 'ambiguous' is ambiguous}}
}
