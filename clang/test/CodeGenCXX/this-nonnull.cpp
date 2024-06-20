// RUN: %clang_cc1 -emit-llvm -o - -triple=x86_64-linux-gnu %s | FileCheck %s -check-prefix=CHECK-YES
// RUN: %clang_cc1 -emit-llvm -o - -fno-delete-null-pointer-checks -triple=x86_64-linux-gnu %s | FileCheck %s -check-prefix=CHECK-NO

struct Struct {
  int many;
  int member;
  int fields;
  void ReturnsVoid();
};

void TestReturnsVoid(Struct &s) {
  s.ReturnsVoid();

  // CHECK-YES: call void @_ZN6Struct11ReturnsVoidEv(ptr noundef nonnull align 4 dereferenceable(12) %0)
  /// FIXME Use dereferenceable after dereferenceable respects NullPointerIsValid.
  // CHECK-NO: call void @_ZN6Struct11ReturnsVoidEv(ptr noundef align 4 dereferenceable_or_null(12) %0)
}

// CHECK-YES: declare void @_ZN6Struct11ReturnsVoidEv(ptr noundef nonnull align 4 dereferenceable(12))
// CHECK-NO: declare void @_ZN6Struct11ReturnsVoidEv(ptr noundef align 4 dereferenceable_or_null(12))
