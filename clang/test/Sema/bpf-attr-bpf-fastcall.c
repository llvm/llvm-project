// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 %s -triple bpf -verify

__attribute__((bpf_fastcall)) int var; // expected-warning {{'bpf_fastcall' attribute only applies to functions and function pointers}}

__attribute__((bpf_fastcall)) void func();
__attribute__((bpf_fastcall(1))) void func_invalid(); // expected-error {{'bpf_fastcall' attribute takes no arguments}}
