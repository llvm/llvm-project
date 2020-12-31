// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefix=LINUX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=DARWIN %s

// Regression test for PR40327

// LINUX: @default_tls ={{.*}} thread_local global i32
// LINUX: @hidden_tls = hidden thread_local global i32
// LINUX: define weak_odr hidden i32* @_ZTW11default_tls()
// LINUX: define weak_odr hidden i32* @_ZTW10hidden_tls()
//
// DARWIN: @default_tls = internal thread_local global i32
// DARWIN: @hidden_tls = internal thread_local global i32
// DARWIN: define cxx_fast_tlscc i32* @_ZTW11default_tls()
// DARWIN: define hidden cxx_fast_tlscc i32* @_ZTW10hidden_tls()

__attribute__((visibility("default"))) thread_local int default_tls;
__attribute__((visibility("hidden"))) thread_local int hidden_tls;
