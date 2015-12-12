namespace ns1 {
struct Base {};
struct Derived : Base {
  Derived() : ns1::Base() {}
};
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: namespaced-base-ctor-init.cpp:4:15: NamespaceRef=ns1:1:11 Extent=[4:15 - 4:18]
// CHECK: namespaced-base-ctor-init.cpp:4:20: TypeRef=struct ns1::Base:2:8 Extent=[4:20 - 4:24]
