// RUN: %clang_cc1 -emit-pch %s -o %t
// RUN: %clang_cc1 -verify -verify-ignore-unexpected=note -include-pch %t -fsyntax-only %s

#ifndef HEADER
#define HEADER

#pragma clang force_cuda_host_device begin
#pragma clang force_cuda_host_device begin
#pragma clang force_cuda_host_device end

void hd1() {}

#else

void hd2() {}

#pragma clang force_cuda_host_device end

void host_only() {}

__attribute__((device)) void device() {
  hd1();
  hd2();
  host_only(); // expected-error {{no matching function for call}}
}

#endif
