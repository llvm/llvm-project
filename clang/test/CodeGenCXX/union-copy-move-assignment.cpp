// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o - | FileCheck %s

union U {
  int a;
  float b;
};

// Odr-use both defaulted assignment operators out of line so their bodies are
// emitted (a trivial assignment at a call site is otherwise memcpy'd directly).
auto get_copy = static_cast<U &(U::*)(const U &)>(&U::operator=);
auto get_move = static_cast<U &(U::*)(U &&)>(&U::operator=);

// Exactly one whole-object memcpy per assignment body.
// CHECK-LABEL: define {{.*}} ptr @_ZN1UaSERKS_
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 4, i1 false)
// CHECK-NOT:     memcpy
// CHECK:         ret ptr

// CHECK-LABEL: define {{.*}} ptr @_ZN1UaSEOS_
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 4, i1 false)
// CHECK-NOT:     memcpy
// CHECK:         ret ptr

union Padded {
  int a;
  char b[5];
};

// sizeof(Padded) == 8, so the whole-object copy includes the tail padding.
auto get_copy_padded = static_cast<Padded &(Padded::*)(const Padded &)>(&Padded::operator=);

// CHECK-LABEL: define {{.*}} ptr @_ZN6PaddedaSERKS_
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)
// CHECK-NOT:     memcpy
// CHECK:         ret ptr

struct WithNamedUnion {
  U u;
  int x;
};

// A named union member is copied as part of the containing class's defaulted
// assignment.
void assign_named(WithNamedUnion *d, const WithNamedUnion *s) { *d = *s; }
// CHECK-LABEL: define {{.*}} @_Z12assign_named
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)

struct WithAnonUnion {
  union {
    int a;
    float b;
  };
  int x;
};

// An anonymous union member is likewise copied.
void assign_anon(WithAnonUnion *d, const WithAnonUnion *s) { *d = *s; }
// CHECK-LABEL: define {{.*}} @_Z11assign_anon
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)
