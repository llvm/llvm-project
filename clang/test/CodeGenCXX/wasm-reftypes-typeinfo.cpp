// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -target-feature +reference-types -emit-llvm -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 %s -triple wasm64-unknown-unknown -target-feature +reference-types -emit-llvm -o - -std=c++11 | FileCheck %s

namespace std {
class type_info;
};

auto &externref = typeid(__externref_t);

// CHECK-DAG: @_ZTSu11externref_t = {{.*}} c"u11externref_t\00", {{.*}}
// CHECK-DAG: @_ZTIu11externref_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11externref_t {{.*}}
