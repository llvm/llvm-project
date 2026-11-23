// Check anonymous namespace deterministic hash generation.
// NOTE: applies to *-msvc targets only.

// RUN: %clang_cc1 -triple=x86_64-pc-windows-msvc -std=c++2a -ast-dump=json -fmacro-prefix-map=%p=/static-path %s | FileCheck %s
// REQUIRES: x86-registered-target

namespace {
    int internal_ns_var = 0;
} 

// The 0xA110234F hash value must be identical on any system with proper path substitution.
// CHECK: "mangledName": "?internal_ns_var@?A0xA110234F@@3HA",
