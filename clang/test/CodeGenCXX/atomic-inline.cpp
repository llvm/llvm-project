// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=x86_64-linux-gnu -target-cpu core2 | FileCheck %s --check-prefix=CORE2
// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=i386-linux-gnu -target-cpu i386 | FileCheck %s --check-prefix=I386
// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=i386-linux-gnu -target-cpu i486 | FileCheck %s --check-prefix=I486
// Check the atomic code generation for cpu targets w/wo cx, cx8 and cx16 support.

struct alignas(4) AM4 {
  short f1, f2;
};
AM4 m4;
AM4 load4() {
  AM4 am;
  // CHECK-LABEL: @_Z5load4v
  // CHECK: load atomic i32, {{.*}} monotonic
  // CORE2-LABEL: @_Z5load4v
  // CORE2: load atomic i32, {{.*}} monotonic
  // I386-LABEL: @_Z5load4v
  // I386: call i32 @__atomic_load_4
  // I486-LABEL: @_Z5load4v
  // I486: load atomic i32, {{.*}} monotonic
  __atomic_load(&m4, &am, 0);
  return am;
}

AM4 s4;
void store4() {
  // CHECK-LABEL: @_Z6store4v
  // CHECK: store atomic i32 {{.*}} monotonic
  // CORE2-LABEL: @_Z6store4v
  // CORE2: store atomic i32 {{.*}} monotonic
  // I386-LABEL: @_Z6store4v
  // I386: call void @__atomic_store_4
  // I486-LABEL: @_Z6store4v
  // I486: store atomic i32 {{.*}} monotonic
  __atomic_store(&m4, &s4, 0);
}

bool cmpxchg4() {
  AM4 am;
  // CHECK-LABEL: @_Z8cmpxchg4v
  // CHECK: cmpxchg i32* {{.*}} monotonic
  // CORE2-LABEL: @_Z8cmpxchg4v
  // CORE2: cmpxchg i32* {{.*}} monotonic
  // I386-LABEL: @_Z8cmpxchg4v
  // I386: call zeroext i1 @__atomic_compare_exchange_4
  // I486-LABEL: @_Z8cmpxchg4v
  // I486: cmpxchg i32* {{.*}} monotonic
  return __atomic_compare_exchange(&m4, &s4, &am, 0, 0, 0);
}

struct alignas(8) AM8 {
  int f1, f2;
};
AM8 m8;
AM8 load8() {
  AM8 am;
  // CHECK-LABEL: @_Z5load8v
  // CHECK: load atomic i64, {{.*}} monotonic
  // CORE2-LABEL: @_Z5load8v
  // CORE2: load atomic i64, {{.*}} monotonic
  // I386-LABEL: @_Z5load8v
  // I386: call i64 @__atomic_load_8
  // I486-LABEL: @_Z5load8v
  // I486: call i64 @__atomic_load_8
  __atomic_load(&m8, &am, 0);
  return am;
}

AM8 s8;
void store8() {
  // CHECK-LABEL: @_Z6store8v
  // CHECK: store atomic i64 {{.*}} monotonic
  // CORE2-LABEL: @_Z6store8v
  // CORE2: store atomic i64 {{.*}} monotonic
  // I386-LABEL: @_Z6store8v
  // I386: call void @__atomic_store_8
  // I486-LABEL: @_Z6store8v
  // I486: call void @__atomic_store_8
  __atomic_store(&m8, &s8, 0);
}

bool cmpxchg8() {
  AM8 am;
  // CHECK-LABEL: @_Z8cmpxchg8v
  // CHECK: cmpxchg i64* {{.*}} monotonic
  // CORE2-LABEL: @_Z8cmpxchg8v
  // CORE2: cmpxchg i64* {{.*}} monotonic
  // I386-LABEL: @_Z8cmpxchg8v
  // I386: call zeroext i1 @__atomic_compare_exchange_8
  // I486-LABEL: @_Z8cmpxchg8v
  // I486: call zeroext i1 @__atomic_compare_exchange_8
  return __atomic_compare_exchange(&m8, &s8, &am, 0, 0, 0);
}

struct alignas(16) AM16 {
  long f1, f2;
};

AM16 m16;
AM16 load16() {
  AM16 am;
  // CHECK-LABEL: @_Z6load16v
  // CHECK: call void @__atomic_load
  // CORE2-LABEL: @_Z6load16v
  // CORE2: load atomic i128, {{.*}} monotonic
  __atomic_load(&m16, &am, 0);
  return am;
}

AM16 s16;
void store16() {
  // CHECK-LABEL: @_Z7store16v
  // CHECK: call void @__atomic_store
  // CORE2-LABEL: @_Z7store16v
  // CORE2: store atomic i128 {{.*}} monotonic
  __atomic_store(&m16, &s16, 0);
}

bool cmpxchg16() {
  AM16 am;
  // CHECK-LABEL: @_Z9cmpxchg16v
  // CHECK: call zeroext i1 @__atomic_compare_exchange
  // CORE2-LABEL: @_Z9cmpxchg16v
  // CORE2: cmpxchg i128* {{.*}} monotonic
  return __atomic_compare_exchange(&m16, &s16, &am, 0, 0, 0);
}
