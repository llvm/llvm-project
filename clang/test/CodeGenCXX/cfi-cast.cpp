// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-derived-cast -fsanitize-trap=cfi-derived-cast -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-DCAST %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-unrelated-cast -fsanitize-trap=cfi-unrelated-cast -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-UCAST %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-unrelated-cast,cfi-cast-strict -fsanitize-trap=cfi-unrelated-cast,cfi-cast-strict -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-UCAST-STRICT %s

// In this test the main thing we are searching for is something like
// 'metadata !"1B"' where "1B" is the mangled name of the class we are
// casting to (or maybe its base class in non-strict mode).

struct A {
  virtual void f();
};

struct B : A {
  virtual void f();
};

struct C : A {};

// CHECK-DCAST-LABEL: define void @_Z3abpP1A
void abp(A *a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.trap()
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  static_cast<B*>(a);
}

// CHECK-DCAST-LABEL: define void @_Z3abrR1A
void abr(A &a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.trap()
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  static_cast<B&>(a);
}

// CHECK-DCAST-LABEL: define void @_Z4abrrO1A
void abrr(A &&a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.trap()
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  static_cast<B&&>(a);
}

// CHECK-UCAST-LABEL: define void @_Z3vbpPv
void vbp(void *p) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.trap()
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  static_cast<B*>(p);
}

// CHECK-UCAST-LABEL: define void @_Z3vbrRc
void vbr(char &r) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.trap()
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  reinterpret_cast<B&>(r);
}

// CHECK-UCAST-LABEL: define void @_Z4vbrrOc
void vbrr(char &&r) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.trap()
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  reinterpret_cast<B&&>(r);
}

// CHECK-UCAST-LABEL: define void @_Z3vcpPv
// CHECK-UCAST-STRICT-LABEL: define void @_Z3vcpPv
void vcp(void *p) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  static_cast<C*>(p);
}

// CHECK-UCAST-LABEL: define void @_Z3bcpP1B
// CHECK-UCAST-STRICT-LABEL: define void @_Z3bcpP1B
void bcp(B *p) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  (C *)p;
}

// CHECK-UCAST-LABEL: define void @_Z8bcp_callP1B
// CHECK-UCAST-STRICT-LABEL: define void @_Z8bcp_callP1B
void bcp_call(B *p) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  ((C *)p)->f();
}
