// RUN: c-index-test -test-load-source all -std=c++20 %s | FileCheck %s

template <class T> struct A {
  struct B;
};

struct C {
  template <class T> friend struct A<T>::B;
};

// CHECK: load-dependent-friend-template.cpp:7:8: StructDecl=C:7:8 (Definition)
// CHECK: load-dependent-friend-template.cpp:8:42: FriendDecl=:8:42
// CHECK-NEXT: load-dependent-friend-template.cpp:8:19: TemplateTypeParameter=T:8:19 (Definition)
// CHECK-NEXT: load-dependent-friend-template.cpp:8:36: TemplateRef=A:3:27
// CHECK-NEXT: load-dependent-friend-template.cpp:8:38: TypeRef=T:8:19
