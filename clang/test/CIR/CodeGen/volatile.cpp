// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int test_load(volatile int *ptr) {
  return *ptr;
}

// CHECK: cir.func @_Z9test_loadPVi
// CHECK:   %{{.+}} = cir.load volatile

void test_store(volatile int *ptr) {
  *ptr = 42;
}

// CHECK: cir.func @_Z10test_storePVi
// CHECK:   cir.store volatile

struct Foo {
  int x;
  volatile int y;
  volatile int z: 4;
};

int test_load_field1(volatile Foo *ptr) {
  return ptr->x;
}

// CHECK: cir.func @_Z16test_load_field1PV3Foo
// CHECK:   %[[MemberAddr:.*]] = cir.get_member
// CHECK:   %{{.+}} = cir.load volatile %[[MemberAddr]]

int test_load_field2(Foo *ptr) {
  return ptr->y;
}

// CHECK: cir.func @_Z16test_load_field2P3Foo
// CHECK:   %[[MemberAddr:.+]] = cir.get_member
// CHECK:   %{{.+}} = cir.load volatile %[[MemberAddr]]

int test_load_field3(Foo *ptr) {
  return ptr->z;
}

// CHECK: cir.func @_Z16test_load_field3P3Foo
// CHECK:   %[[MemberAddr:.+]] = cir.get_member
// CHECK:   %{{.+}} = cir.load volatile %[[MemberAddr]]

void test_store_field1(volatile Foo *ptr) {
  ptr->x = 42;
}

// CHECK: cir.func @_Z17test_store_field1PV3Foo
// CHECK:   %[[MemberAddr:.+]] = cir.get_member
// CHECK:   cir.store volatile %{{.+}}, %[[MemberAddr]]

void test_store_field2(Foo *ptr) {
  ptr->y = 42;
}

// CHECK: cir.func @_Z17test_store_field2P3Foo
// CHECK:   %[[MemberAddr:.+]] = cir.get_member
// CHECK:   cir.store volatile %{{.+}}, %[[MemberAddr]]

void test_store_field3(Foo *ptr) {
  ptr->z = 4;
}

// CHECK: cir.func @_Z17test_store_field3P3Foo
// CHECK:   %[[MemberAddr:.+]] = cir.get_member
// CHECK:   cir.store volatile %{{.+}}, %[[MemberAddr]]
