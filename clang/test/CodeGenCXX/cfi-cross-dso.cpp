// RUN: %clang_cc1 -flto -triple x86_64-unknown-linux -fsanitize=cfi-vcall -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -flto -triple x86_64-pc-windows-msvc -fsanitize=cfi-vcall  -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

struct A {
  A();
  virtual void f();
};

A::A() {}
void A::f() {}

void caller(A* a) {
  a->f();
}

namespace {
struct B {
  virtual void f();
};

void B::f() {}
} // namespace

void g() {
  B b;
  b.f();
}

// MS: @[[B_VTABLE:.*]] = private unnamed_addr constant [2 x i8*] {{.*}}@"\01??_R4B@?A@@6B@"{{.*}}@"\01?f@B@?A@@UEAAXXZ"

// CHECK:   %[[VT:.*]] = load void (%struct.A*)**, void (%struct.A*)***
// CHECK:   %[[VT2:.*]] = bitcast {{.*}}%[[VT]] to i8*, !nosanitize
// ITANIUM:   %[[TEST:.*]] = call i1 @llvm.type.test(i8* %[[VT2]], metadata !"_ZTS1A"), !nosanitize
// MS:   %[[TEST:.*]] = call i1 @llvm.type.test(i8* %[[VT2]], metadata !"?AUA@@"), !nosanitize
// CHECK:   br i1 %[[TEST]], label %[[CONT:.*]], label %[[SLOW:.*]], {{.*}} !nosanitize
// CHECK: [[SLOW]]
// ITANIUM:   call void @__cfi_slowpath_diag(i64 7004155349499253778, i8* %[[VT2]], {{.*}}) {{.*}} !nosanitize
// MS:   call void @__cfi_slowpath_diag(i64 -8005289897957287421, i8* %[[VT2]], {{.*}}) {{.*}} !nosanitize
// CHECK:   br label %[[CONT]], !nosanitize
// CHECK: [[CONT]]
// CHECK:   call void %{{.*}}(%struct.A* %{{.*}})

// No hash-based bit set entry for (anonymous namespace)::B
// ITANIUM-NOT: !{i64 {{.*}}, [3 x i8*]* @_ZTVN12_GLOBAL__N_11BE,
// MS-NOT: !{i64 {{.*}}, [2 x i8*]* @[[B_VTABLE]],
