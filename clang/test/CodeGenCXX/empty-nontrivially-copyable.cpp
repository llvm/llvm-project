// RUN: %clang_cc1 -no-opaque-pointers -triple armv7-apple-ios -x c++ -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios -x c++ -emit-llvm -o - %s | FileCheck %s

// According to the Itanium ABI (3.1.1), types with non-trivial copy
// constructors passed by value should be passed indirectly, with the caller
// creating a temporary.

struct Empty;

struct Empty {
  Empty(const Empty &e);
  bool check();
};

bool foo(Empty e) {
// CHECK: @_Z3foo5Empty(%struct.Empty* noundef %e)
// CHECK: call {{.*}} @_ZN5Empty5checkEv(%struct.Empty* {{[^,]*}} %e)
  return e.check();
}

void caller(Empty &e) {
// CHECK: @_Z6callerR5Empty(%struct.Empty* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %e)
// CHECK: call {{.*}} @_ZN5EmptyC1ERKS_(%struct.Empty* {{[^,]*}} [[NEWTMP:%.*]], %struct.Empty*
// CHECK: call {{.*}} @_Z3foo5Empty(%struct.Empty* noundef [[NEWTMP]])
  foo(e);
}
