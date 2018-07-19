// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-win32 | FileCheck -check-prefix X64 %s

namespace ClassScopeSpecialization {
  struct Type {
    template <int i>
    void Foo() {}
    template <>
    void Foo<0>() {}
  };

  void call() {
    Type T;
// CHECK: call {{.*}} @"\01??$Foo@$0A@@Type@ClassScopeSpecialization@@QAEXXZ"
// X64: call {{.*}} @"\01??$Foo@$0A@@Type@ClassScopeSpecialization@@QEAAXXZ"
    T.Foo<0>();
  }
}
