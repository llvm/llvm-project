// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
struct A { virtual void f(); };
struct B : A { };

// CHECK: {{define.*@_Z1fP1A}}
// CHECK-SAME:  personality ptr @__gxx_personality_v0
B fail;
const B& f(A *a) {
  try {
    // CHECK: call ptr @__dynamic_cast
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_cast() [[NR:#[0-9]+]]
    dynamic_cast<const B&>(*a);
  } catch (...) {
    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   catch ptr null
  }
  return fail;
}

// CHECK: declare ptr @__dynamic_cast(ptr, ptr, ptr, i64) [[NUW_RO:#[0-9]+]]

// CHECK: attributes [[NUW_RO]] = { nounwind willreturn memory(read) }
// CHECK: attributes [[NR]] = { noreturn }
