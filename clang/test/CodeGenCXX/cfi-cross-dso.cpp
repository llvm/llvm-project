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

// MS: @[[B_VTABLE:.*]] = private unnamed_addr constant { [2 x ptr] } {{.*}}@"??_R4B@?A0x{{[^@]*}}@@6B@"{{.*}}@"?f@B@?A0x{{[^@]*}}@@UEAAXXZ"

// CHECK-LABEL: caller
// CHECK:   %[[A:.*]] = load ptr, ptr
// CHECK:   %[[VT:.*]] = load ptr, ptr %[[A]]
// ITANIUM: %[[TEST:.*]] = call i1 @llvm.type.test(ptr %[[VT]], metadata !"_ZTS1A"), !nosanitize
// MS:      %[[TEST:.*]] = call i1 @llvm.type.test(ptr %[[VT]], metadata !"?AUA@@"), !nosanitize
// CHECK:   br i1 %[[TEST]], label %[[CONT:.*]], label %[[SLOW:.*]], {{.*}} !nosanitize
// CHECK: [[SLOW]]
// ITANIUM: call void @__cfi_slowpath_diag(i64 7004155349499253778, ptr %[[VT]], {{.*}}) {{.*}} !nosanitize
// MS:      call void @__cfi_slowpath_diag(i64 -8005289897957287421, ptr %[[VT]], {{.*}}) {{.*}} !nosanitize
// CHECK:   br label %[[CONT]], !nosanitize
// CHECK: [[CONT]]
// CHECK:   call void %{{.*}}(ptr {{[^,]*}} %{{.*}})

// No hash-based bit set entry for (anonymous namespace)::B
// ITANIUM-NOT: !{i64 {{.*}}, ptr @_ZTVN12_GLOBAL__N_11BE,
// MS-NOT: !{i64 {{.*}}, ptr @[[B_VTABLE]],
