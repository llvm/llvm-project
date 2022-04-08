// RUN: %clang_cc1 -no-opaque-pointers -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
struct A { virtual void f(); };
struct B : A { };

// CHECK: {{define.*@_Z1fP1A}}
// CHECK-SAME:  personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
B fail;
const B& f(A *a) {
  try {
    // CHECK: call i8* @__dynamic_cast
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_cast() [[NR:#[0-9]+]]
    dynamic_cast<const B&>(*a);
  } catch (...) {
    // CHECK:      landingpad { i8*, i32 }
    // CHECK-NEXT:   catch i8* null
  }
  return fail;
}

// CHECK: declare i8* @__dynamic_cast(i8*, i8*, i8*, i64) [[NUW_RO:#[0-9]+]]

// CHECK: attributes [[NUW_RO]] = { nounwind readonly }
// CHECK: attributes [[NR]] = { noreturn }
