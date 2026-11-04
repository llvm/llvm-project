// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - | FileCheck %s

class Empty {
};

class Long : public Empty {
public:
  long l;
};

// CHECK: define{{.*}} i64 @_Z4foo15Empty(i64 %e.coerce)
Empty foo1(Empty e) {
  return e;
}

// CHECK: define{{.*}} %class.Long @_Z4foo24Long(i64 %l.coerce)
Long foo2(Long l) {
  return l;
}

// CHECK: define{{.*}} i64 @_Z4foo34Long(i64 %l.coerce)
long foo3(Long l) {
  return l.l;
}
