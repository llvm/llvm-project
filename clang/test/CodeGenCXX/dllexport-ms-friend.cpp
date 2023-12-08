// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -emit-llvm -O0 -o - %s | FileCheck %s

// Friend functions defined in classes are emitted.
// CHECK: define weak_odr dso_local dllexport void @"?friend1@@YAXXZ"()
struct FuncFriend1 {
  friend __declspec(dllexport) void friend1() {}
};

// But function templates and functions defined in class templates are not
// emitted.
// CHECK-NOT: friend2
// CHECK-NOT: friend3
// CHECK-NOT: friend4
struct FuncFriend2 {
  template<typename> friend __declspec(dllexport) void friend2() {}
};
template<typename> struct FuncFriend3 {
  friend __declspec(dllexport) void friend3() {}
  struct Inner {
    friend __declspec(dllexport) void friend4() {}
  };
};
