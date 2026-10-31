// RUN: %clang_cc1 -triple riscv32-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +zve64d | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +zve64d | FileCheck %s

// The psABI requires a function using a standard calling convention variant to
// append an ABI tag to its mangled name.

// CHECK-DAG: @_Z14default_vls_ccB16riscv_vls_cc_128v
__attribute__((riscv_vls_cc)) void default_vls_cc() {}

// CHECK-DAG: @_Z7vlen_32B15riscv_vls_cc_32v
__attribute__((riscv_vls_cc(32))) void vlen_32() {}

// CHECK-DAG: @_Z8vlen_128B16riscv_vls_cc_128v
__attribute__((riscv_vls_cc(128))) void vlen_128() {}

// CHECK-DAG: @_Z10vlen_65536B18riscv_vls_cc_65536v
__attribute__((riscv_vls_cc(65536))) void vlen_65536() {}

// The ABI_VLEN is part of the tag, so functions that differ only in ABI_VLEN
// get different mangled names.
// CHECK-DAG: @_ZN3v641fB15riscv_vls_cc_64Ev
namespace v64 { __attribute__((riscv_vls_cc(64))) void f() {} }
// CHECK-DAG: @_ZN4v5121fB16riscv_vls_cc_512Ev
namespace v512 { __attribute__((riscv_vls_cc(512))) void f() {} }

// The calling convention can also come from a typedef.
typedef void vls_fn_t(void) __attribute__((riscv_vls_cc(256)));
// CHECK-DAG: @_Z12from_typedefB16riscv_vls_cc_256v
vls_fn_t from_typedef;
void from_typedef() {}

// The tag is sorted together with an explicit abi_tag.
// CHECK-DAG: @_Z6taggedB16riscv_vls_cc_128B4userv
__attribute__((abi_tag("user"), riscv_vls_cc(128))) void tagged() {}

// One tag sorts before it and one after, and they are written the other way
// round.
// CHECK-DAG: @_Z7tagged2B3abcB16riscv_vls_cc_128B4userv
__attribute__((abi_tag("user", "abc"), riscv_vls_cc(128))) void tagged2() {}

namespace ns {
struct S {
  // CHECK-DAG: @_ZN2ns1S6memberB16riscv_vls_cc_128Ev
  __attribute__((riscv_vls_cc(128))) void member() {}
};
void instantiate() { S().member(); }
} // namespace ns

// CHECK-DAG: @_Z9template_B16riscv_vls_cc_128IiEvT_
template <typename T> __attribute__((riscv_vls_cc(128))) void template_(T) {}
template void template_<int>(int);

// The VLA vector calling convention variant has no ABI tag.
// CHECK-DAG: @_Z9vector_ccv
__attribute__((riscv_vector_cc)) void vector_cc() {}

// CHECK-DAG: @"_ZNK3$_0clB16riscv_vls_cc_128Ev"
auto lam = []() __attribute__((riscv_vls_cc(128))) {};
void use_lam() { lam(); }

// CHECK-DAG: @_Z5plainv
void plain() {}
