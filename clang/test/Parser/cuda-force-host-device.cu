// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check the force_cuda_host_device pragma.

#pragma clang force_cuda_host_device begin
void f();
#pragma clang force_cuda_host_device begin
void g();
#pragma clang force_cuda_host_device end
void h();
#pragma clang force_cuda_host_device end

void i(); // expected-note {{not viable}}

void host() {
  f();
  g();
  h();
  i();
}

__attribute__((device)) void device() {
  f();
  g();
  h();
  i(); // expected-error {{no matching function}}
}

#pragma clang force_cuda_host_device foo
// expected-warning@-1 {{incorrect use of #pragma clang force_cuda_host_device begin|end}}

#pragma clang force_cuda_host_device
// expected-warning@-1 {{incorrect use of #pragma clang force_cuda_host_device begin|end}}

#pragma clang force_cuda_host_device begin foo
// expected-warning@-1 {{incorrect use of #pragma clang force_cuda_host_device begin|end}}
