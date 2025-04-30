// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

#define __global__ __attribute__((global))

__attribute__((reqd_work_group_size(0x100000000, 1, 1))) // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__global__ void TestTooBigArg1(void);

__attribute__((work_group_size_hint(0x100000000, 1, 1))) // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__global__ void TestTooBigArg2(void);

template <int... Args>
__attribute__((reqd_work_group_size(Args))) // expected-error {{expression contains unexpanded parameter pack 'Args'}}
__global__ void TestTemplateVariadicArgs1(void) {}

template <int... Args>
__attribute__((work_group_size_hint(Args))) // expected-error {{expression contains unexpanded parameter pack 'Args'}}
__global__ void TestTemplateVariadicArgs2(void) {}

template <class a> // expected-note {{declared here}}
__attribute__((reqd_work_group_size(a, 1, 1))) // expected-error {{'a' does not refer to a value}}
__global__ void TestTemplateArgClass1(void) {}

template <class a> // expected-note {{declared here}}
__attribute__((work_group_size_hint(a, 1, 1))) // expected-error {{'a' does not refer to a value}}
__global__ void TestTemplateArgClass2(void) {}

constexpr int A = 512;

__attribute__((reqd_work_group_size(A, A, A)))
__global__ void TestConstIntArg1(void) {}

__attribute__((work_group_size_hint(A, A, A)))
__global__ void TestConstIntArg2(void) {}

int B = 512;
__attribute__((reqd_work_group_size(B, 1, 1))) // expected-error {{attribute requires parameter 0 to be an integer constant}}
__global__ void TestNonConstIntArg1(void) {}

__attribute__((work_group_size_hint(B, 1, 1))) // expected-error {{attribute requires parameter 0 to be an integer constant}}
__global__ void TestNonConstIntArg2(void) {}

constexpr int C = -512;
__attribute__((reqd_work_group_size(C, 1, 1))) // expected-error {{attribute requires a non-negative integral compile time constant expression}}
__global__ void TestNegativeConstIntArg1(void) {}

__attribute__((work_group_size_hint(C, 1, 1))) // expected-error {{attribute requires a non-negative integral compile time constant expression}}
__global__ void TestNegativeConstIntArg2(void) {}


__attribute__((reqd_work_group_size(A, 0, 1))) // expected-error {{attribute must be greater than 0}}
__global__ void TestZeroArg1(void) {}

__attribute__((work_group_size_hint(A, 0, 1))) // expected-error {{attribute must be greater than 0}}
__global__ void TestZeroArg2(void) {}



