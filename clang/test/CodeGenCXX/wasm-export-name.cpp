// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

namespace ns {
  int __attribute__((export_name)) namespaced_var = 10;
  void __attribute__((export_name)) namespaced_func() {}
}

int __attribute__((export_name)) overloaded(int x) { return x; }
int __attribute__((export_name)) overloaded(double x) { return (int)x; }

extern "C" {
  int __attribute__((export_name)) extern_c_var = 20;
  void __attribute__((export_name)) extern_c_func() {}
}

// CHECK: @_ZN2ns14namespaced_varE = global i32 10, align 4 [[VAR_NS:#[0-9]+]]
// CHECK: @extern_c_var = global i32 20, align 4 [[VAR_EXTERN_C:#[0-9]+]]

// CHECK: define void @_ZN2ns15namespaced_funcEv() [[FN_NS:#[0-9]+]]
// CHECK: define {{.*}}i32 @_Z10overloadedi({{.*}}) [[FN_OVI:#[0-9]+]]
// CHECK: define {{.*}}i32 @_Z10overloadedd({{.*}}) [[FN_OVD:#[0-9]+]]
// CHECK: define void @extern_c_func() [[FN_EXTERN_C:#[0-9]+]]

// CHECK: attributes [[VAR_NS]] = { "wasm-export-name"="_ZN2ns14namespaced_varE" }
// CHECK: attributes [[VAR_EXTERN_C]] = { "wasm-export-name"="extern_c_var" }
// CHECK: attributes [[FN_NS]] = { {{.*}}"wasm-export-name"="_ZN2ns15namespaced_funcEv" }
// CHECK: attributes [[FN_OVI]] = { {{.*}}"wasm-export-name"="_Z10overloadedi" }
// CHECK: attributes [[FN_OVD]] = { {{.*}}"wasm-export-name"="_Z10overloadedd" }
// CHECK: attributes [[FN_EXTERN_C]] = { {{.*}}"wasm-export-name"="extern_c_func" }
