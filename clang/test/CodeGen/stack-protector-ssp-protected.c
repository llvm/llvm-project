// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 2 | FileCheck %s --check-prefix=STRONG
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 3 | FileCheck %s --check-prefix=STRONG

// Character array: protected under -fstack-protector and -fstack-protector-strong.
// SSP-LABEL: @char_arr(
// SSP: call void @llvm.ssp.protected(ptr %buf)
// STRONG-LABEL: @char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %buf)
void char_arr(void) { char buf[8]; (void)buf; }

// Integer array: not protected under -fstack-protector (not a char array),
// but protected under -fstack-protector-strong (all arrays).
// SSP-LABEL: @int_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @int_arr(
// STRONG: call void @llvm.ssp.protected(ptr %arr)
void int_arr(void) { int arr[4]; (void)arr; }

// Struct containing a character array: protected under both modes.
struct S { char buf[8]; };
// SSP-LABEL: @struct_char(
// SSP: call void @llvm.ssp.protected(ptr %s)
// STRONG-LABEL: @struct_char(
// STRONG: call void @llvm.ssp.protected(ptr %s)
void struct_char(void) { struct S s; (void)s; }

// VLA: the alloca is variable-length, so it should be assumed to be large
// SSP-LABEL: @vla_char(
// SSP: call void @llvm.ssp.protected
// STRONG-LABEL: @vla_char(
// STRONG: call void @llvm.ssp.protected
void vla_char(int n) { char arr[n]; (void)arr; }

// VLA: the alloca is variable-length, so it should be assumed to be large
// SSP-LABEL: @vla_int(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @vla_int(
// STRONG: call void @llvm.ssp.protected
void vla_int(int n) { int arr[n]; (void)arr; }

// __builtin_alloca with variable size: always protected (LangRef: "variable
// lengths").
// SSP-LABEL: @alloca_var(
// SSP: call void @llvm.ssp.protected(ptr %{{.*}})
// STRONG-LABEL: @alloca_var(
// STRONG: call void @llvm.ssp.protected(ptr %{{.*}})
void alloca_var(int n) { void *p = __builtin_alloca(n); (void)p; }

// __builtin_alloca with constant size >= SSPBufferSize (default 8): protected
// under both ssp and sspstrong (LangRef: "lengths larger than SSPBufferSize").
// SSP-LABEL: @alloca_large(
// SSP: call void @llvm.ssp.protected(ptr %{{.*}})
// STRONG-LABEL: @alloca_large(
// STRONG: call void @llvm.ssp.protected(ptr %{{.*}})
void alloca_large(void) { void *p = __builtin_alloca(8); (void)p; }

// __builtin_alloca with constant size < SSPBufferSize: not protected under ssp,
// but protected under sspstrong (LangRef: "any call to alloca regardless of
// size").
// SSP-LABEL: @alloca_small(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @alloca_small(
// STRONG: call void @llvm.ssp.protected(ptr %{{.*}})
void alloca_small(void) { void *p = __builtin_alloca(4); (void)p; }
