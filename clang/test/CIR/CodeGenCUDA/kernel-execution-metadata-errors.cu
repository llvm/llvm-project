// RUN: %clang_cc1 %s -x cuda -fcuda-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir -verify -o %t
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -triple spirv64-amd-amdhsa -fclangir -emit-cir -verify -o %t

typedef bool bool4 __attribute__((ext_vector_type(4)));
// expected-error@+1 {{ClangIR code gen Not Yet Implemented: vec_type_hint with a boolean vector type}}
__attribute__((global, vec_type_hint(bool4))) void kernel() {}
