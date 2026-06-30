// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=X64

typedef float __m128 __attribute__((__vector_size__(16)));

// HVA with base class
struct base_hva { __m128 v; };
struct test_hva : base_hva { test_hva(double); protected: __m128 v2; };

// HFA with base class
struct base_hfa { double v; };
struct test_hfa : base_hfa { test_hfa(double); protected: double v2; };

// 1. Vectorcall returns should be direct (not sret)
test_hva __vectorcall ret_hva_vectorcall(test_hva *x) { return *x; }
// X86-LABEL: define dso_local x86_vectorcallcc %struct.test_hva @"?ret_hva_vectorcall@@YQ?AUtest_hva@@PAU1@@Z"(ptr inreg noundef %x)
// X64-LABEL: define dso_local x86_vectorcallcc %struct.test_hva @"?ret_hva_vectorcall@@YQ?AUtest_hva@@PEAU1@@Z"(ptr noundef %x)

test_hfa __vectorcall ret_hfa_vectorcall(test_hfa *x) { return *x; }
// X86-LABEL: define dso_local x86_vectorcallcc %struct.test_hfa @"?ret_hfa_vectorcall@@YQ?AUtest_hfa@@PAU1@@Z"(ptr inreg noundef %x)
// X64-LABEL: define dso_local x86_vectorcallcc %struct.test_hfa @"?ret_hfa_vectorcall@@YQ?AUtest_hfa@@PEAU1@@Z"(ptr noundef %x)

// 2. Cdecl returns should be indirect (sret) because they are not aggregates
test_hva __cdecl ret_hva_cdecl(test_hva *x) { return *x; }
// X86-LABEL: define dso_local void @"?ret_hva_cdecl@@YA?AUtest_hva@@PAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.test_hva) align 16 %agg.result, ptr noundef %x)
// X64-LABEL: define dso_local void @"?ret_hva_cdecl@@YA?AUtest_hva@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.test_hva) align 16 %agg.result, ptr noundef %x)

test_hfa __cdecl ret_hfa_cdecl(test_hfa *x) { return *x; }
// X86-LABEL: define dso_local void @"?ret_hfa_cdecl@@YA?AUtest_hfa@@PAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.test_hfa) align 8 %agg.result, ptr noundef %x)
// X64-LABEL: define dso_local void @"?ret_hfa_cdecl@@YA?AUtest_hfa@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.test_hfa) align 8 %agg.result, ptr noundef %x)
