/// Tests that SPIRV64 warns on non-SPIR-V calling conventions without a host.

// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown \
// RUN:   -fsyntax-only -verify %s

void __vectorcall vector_func(float x, float y) {} // expected-warning {{'__vectorcall' calling convention is not supported for this target}}
void default_func(int x) {}
