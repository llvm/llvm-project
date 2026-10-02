// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexceptions -fcxx-exceptions -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=EXC

// CIRGen serializes the LangOptions facts that post-CIRGen lowering consumes
// onto the ModuleOp as a #cir.lowering_lang_options attribute, unconditionally, so
// a reloaded .cir module lowers the same way it was compiled without a live
// clang::LangOptions.

int x;

// Plain C++: exceptions off, thread-safe statics on, no CUDA/HIP/OpenMP. The
// clang_abi_compat ordinal mirrors LangOptions::ClangABI and is left as a
// regex since its default (Latest) shifts with each Clang release.
// PLAIN: cir.lowering_lang_options = #cir.lowering_lang_options<
// PLAIN-SAME: exceptions = false
// PLAIN-SAME: threadsafe_statics = true
// PLAIN-SAME: cuda = false
// PLAIN-SAME: cuda_is_device = false
// PLAIN-SAME: hip = false
// PLAIN-SAME: gpu_rdc = false
// PLAIN-SAME: openmp = false
// PLAIN-SAME: openmp_is_target_device = false
// PLAIN-SAME: clang_abi_compat = {{[0-9]+}}>

// -fexceptions -fcxx-exceptions flips exceptions. (OpenMP is exercised in the
// #cir.lowering_lang_options round-trip test, since -fopenmp on a global here hits
// an unrelated CIRGen NYI for OpenMP globals.)
// EXC: cir.lowering_lang_options = #cir.lowering_lang_options<
// EXC-SAME: exceptions = true
// EXC-SAME: threadsafe_statics = true
// EXC-SAME: openmp = false
// EXC-SAME: openmp_is_target_device = false
// EXC-SAME: clang_abi_compat = {{[0-9]+}}>
