// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

namespace test1 {
struct S {
  double x;
};

typedef struct S __attribute__((__aligned__(2))) SS;

SS ss;

// CHECK: @{{.*}}test1{{.*}}ss{{.*}} = global %"struct.test1::S" zeroinitializer, align 2
} // namespace test1

namespace test2 {
struct __attribute__((__aligned__(2))) S {
  double x;
};

typedef struct S SS;

SS ss;

// CHECK: @{{.*}}test2{{.*}}ss{{.*}} = global %"struct.test2::S" zeroinitializer, align 8
} // namespace test2
