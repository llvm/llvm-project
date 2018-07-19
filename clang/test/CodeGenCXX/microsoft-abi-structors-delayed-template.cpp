// RUN: %clang_cc1 -emit-llvm -fdelayed-template-parsing -std=c++11 -o - -triple=i386-pc-win32 %s > %t
// RUN: FileCheck %s < %t

// PR20671
namespace vtable_referenced_from_template {
struct ImplicitCtor {
  virtual ~ImplicitCtor();
};
template <class T> void foo(T t) { new ImplicitCtor; }
void bar() { foo(0); }
// CHECK: store {{.*}} @"\01??_7ImplicitCtor@vtable_referenced_from_template@@6B@"
}
