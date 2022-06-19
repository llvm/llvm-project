// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

namespace test1 {
struct __attribute__((__aligned__(2))) S {
  double d;
};

S s;

// CHECK: @{{.*}}test1{{.*}}s{{.*}} = global %"struct.test1::S" zeroinitializer, align 8
} // namespace test1

namespace test2 {
struct __attribute__((__aligned__(2), packed)) S {
  double d;
};

S s;

// CHECK: @{{.*}}test2{{.*}}s{{.*}} = global %"struct.test2::S" zeroinitializer, align 2
} // namespace test2

namespace test3 {
struct __attribute__((__aligned__(16))) S {
  double d;
};

S s;

// CHECK: @{{.*}}test3{{.*}}s{{.*}} = global %"struct.test3::S" zeroinitializer, align 16
} // namespace test3

namespace test4 {
struct __attribute__((aligned(2))) SS {
  double d;
};

struct S {
  struct SS ss;
} s;

// CHECK: @{{.*}}test4{{.*}}s{{.*}} = global %"struct.test4::S" zeroinitializer, align 8
} // namespace test4

namespace test5 {
struct __attribute__((aligned(2), packed)) SS {
  double d;
};

struct S {
  struct SS ss;
} s;

// CHECK: @{{.*}}test5{{.*}}s{{.*}} = global %"struct.test5::S" zeroinitializer, align 2
} // namespace test5
