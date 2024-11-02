// RUN: %clang_cc1 -triple x86_64-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-darwin -std=c++11 -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=EXCEPTIONS

// PR36748
// rdar://problem/45805151

// Classes to verify order of destroying function parameters.
struct S1 {
  ~S1();
};
struct S2 {
  ~S2();
};

struct Base {
  // Use variadic args to cause inlining the inherited constructor.
  Base(const S1&, const S2&, const char *fmt, ...) {}
};

struct NonTrivialDtor {
  ~NonTrivialDtor() {}
};
struct Inheritor : public NonTrivialDtor, public Base {
  using Base::Base;
};

void f() {
  Inheritor(S1(), S2(), "foo");
  // CHECK-LABEL: define{{.*}} void @_Z1fv
  // CHECK: %[[TMP1:.*]] = alloca %struct.S1
  // CHECK: %[[TMP2:.*]] = alloca %struct.S2
  // CHECK: call void (ptr, ptr, ptr, ptr, ...) @_ZN4BaseC2ERK2S1RK2S2PKcz(ptr {{.*}}, ptr noundef nonnull align 1 dereferenceable(1) %[[TMP1]], ptr noundef nonnull align 1 dereferenceable(1) %[[TMP2]], ptr {{.*}})
  // CHECK-NEXT: call void @_ZN9InheritorD1Ev(ptr {{.*}})
  // CHECK-NEXT: call void @_ZN2S2D1Ev(ptr {{[^,]*}} %[[TMP2]])
  // CHECK-NEXT: call void @_ZN2S1D1Ev(ptr {{[^,]*}} %[[TMP1]])

  // EXCEPTIONS-LABEL: define{{.*}} void @_Z1fv
  // EXCEPTIONS: %[[TMP1:.*]] = alloca %struct.S1
  // EXCEPTIONS: %[[TMP2:.*]] = alloca %struct.S2
  // EXCEPTIONS: invoke void (ptr, ptr, ptr, ptr, ...) @_ZN4BaseC2ERK2S1RK2S2PKcz(ptr {{.*}}, ptr noundef nonnull align 1 dereferenceable(1) %[[TMP1]], ptr noundef nonnull align 1 dereferenceable(1) %[[TMP2]], ptr {{.*}})
  // EXCEPTIONS-NEXT: to label %[[CONT:.*]] unwind label %[[LPAD:.*]]

  // EXCEPTIONS: [[CONT]]:
  // EXCEPTIONS-NEXT: call void @_ZN9InheritorD1Ev(ptr {{.*}})
  // EXCEPTIONS-NEXT: call void @_ZN2S2D1Ev(ptr {{[^,]*}} %[[TMP2]])
  // EXCEPTIONS-NEXT: call void @_ZN2S1D1Ev(ptr {{[^,]*}} %[[TMP1]])

  // EXCEPTIONS: [[LPAD]]:
  // EXCEPTIONS: call void @_ZN14NonTrivialDtorD2Ev(ptr {{.*}})
  // EXCEPTIONS-NEXT: call void @_ZN2S2D1Ev(ptr {{[^,]*}} %[[TMP2]])
  // EXCEPTIONS-NEXT: call void @_ZN2S1D1Ev(ptr {{[^,]*}} %[[TMP1]])
}
