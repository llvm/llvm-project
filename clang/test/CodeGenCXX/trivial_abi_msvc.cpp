// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++11 -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_TRIVIAL:.*]] = type { ptr }
struct __attribute__((trivial_abi)) Trivial {
  int *p;
  Trivial() : p(0) {}
  Trivial(const Trivial &) noexcept = default;
};

// CHECK-LABEL: define{{.*}} i64 @"?retTrivial@@YA?AUTrivial@@XZ"(
// CHECK: %retval = alloca %[[STRUCT_TRIVIAL]], align 8
// CHECK: %call = call noundef ptr @"??0Trivial@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %retval)
// CHECK: %coerce.dive = getelementptr inbounds %[[STRUCT_TRIVIAL]], ptr %retval, i32 0, i32 0
// CHECK: %0 = load ptr, ptr %coerce.dive, align 8
// CHECK: %coerce.val.pi = ptrtoint ptr %0 to i64
// CHECK: ret i64 %coerce.val.pi
Trivial retTrivial() {
  Trivial s;
  return s;
}

struct TrivialInstance {
    Trivial instanceMethod();
    static Trivial staticMethod();
};

// We need to make sure that instanceMethod has a sret return value since `this` will always go in the register.
// CHECK-LABEL: define{{.*}} void @"?instanceMethod@TrivialInstance@@QEAA?AUTrivial@@XZ"({{.*}} sret(%struct.Trivial{{.*}}
Trivial TrivialInstance::instanceMethod() { return {}; }
// CHECK-LABEL: define{{.*}} i64 @"?staticMethod@TrivialInstance@@SA?AUTrivial@@XZ"(
Trivial TrivialInstance::staticMethod() { return {}; }
