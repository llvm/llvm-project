// RUN: %clang_cc1 -std=c++11 -fsanitize=vptr -emit-llvm %s -o - -triple wasm32-unknown-emscripten | FileCheck %s

struct S {
  virtual ~S() {}
  int a;
};

struct T : S {
  int b;
};

// CHECK-LABEL: @_Z15bad_static_castv
void bad_static_cast() {
  S s;
  // CHECK: br i1 %[[NONNULL:.*]], label %[[CONT:.*]], label %[[MISS:.*]], !prof
  // CHECK: [[MISS]]:
  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss_abort
  // CHECK: [[CONT]]:
  T &r = static_cast<T &>(s);
}
