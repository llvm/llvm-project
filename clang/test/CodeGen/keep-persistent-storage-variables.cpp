// RUN: %clang_cc1 -fkeep-persistent-storage-variables -emit-llvm %s -o - -triple=x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fkeep-persistent-storage-variables -emit-llvm %s -o - -triple=powerpc64-ibm-aix-xcoff | FileCheck %s

// CHECK: @_ZL2g1 = internal global i32 0, align 4
// CHECK: @_ZL2g2 = internal global i32 1, align 4
// CHECK: @tl1 = thread_local global i32 0, align 4
// CHECK: @tl2 = thread_local global i32 3, align 4
// CHECK: @_ZL3tl3 = internal thread_local global i32 0, align 4
// CHECK: @_ZL3tl4 = internal thread_local global i32 4, align 4
// CHECK: @g5 = global i32 0, align 4
// CHECK: @g6 = global i32 6, align 4
// CHECK: @_ZZ5test3vE2s3 = internal global i32 0, align 4
// CHECK: @_ZN12_GLOBAL__N_12s4E = internal global i32 42, align 4
// CHECK: @_ZZ5test5vE3tl5 = internal thread_local global i32 1, align 4
// CHECK: @_ZN2ST2s6E = global i32 7, align 4
// CHECK: @_Z2v7 = internal global %union.anon zeroinitializer, align 4
// CHECK: @_ZDC2v8E = global %struct.ST8 zeroinitializer, align 4
// CHECK: @llvm{{(\.compiler)?}}.used = appending global [14 x ptr] [ptr @_ZL2g1, ptr @_ZL2g2, ptr @tl1, ptr @tl2, ptr @_ZL3tl3, ptr @_ZL3tl4, ptr @g5, ptr @g6, ptr @_ZZ5test3vE2s3, ptr @_ZN12_GLOBAL__N_12s4E, ptr @_ZZ5test5vE3tl5, ptr @_ZN2ST2s6E, ptr @_Z2v7, ptr @_ZDC2v8E], section "llvm.metadata"

static int g1;
static int g2 = 1;
__thread int tl1;
__thread int tl2 = 3;
static __thread int tl3;
static __thread int tl4 = 4;
int g5;
int g6 = 6;

int test3() {
  static int s3 = 0;
  ++s3;
  return s3;
}

namespace {
  int s4 = 42;
}

int test5() {
  static __thread int tl5 = 1;
  ++tl5;
  return tl5;
}

struct ST {
  static int s6;
};
int ST::s6 = 7;

static union { int v7; };

struct ST8 { int v8; };
auto [v8] = ST8{0};
