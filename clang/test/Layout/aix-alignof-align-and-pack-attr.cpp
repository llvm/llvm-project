// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

namespace test1 {
struct __attribute__((__aligned__(2))) C {
  double x;
} c;

// CHECK: @{{.*}}test1{{.*}}c{{.*}} = global %"struct.test1::C" zeroinitializer, align 8
} // namespace test1

namespace test2 {
struct __attribute__((__aligned__(2), packed)) C {
  double x;
} c;

// CHECK: @{{.*}}test2{{.*}}c{{.*}} = global %"struct.test2::C" zeroinitializer, align 2
} // namespace test2

namespace test3 {
struct __attribute__((__aligned__(16))) C {
  double x;
} c;

// CHECK: @{{.*}}test3{{.*}}c{{.*}} = global %"struct.test3::C" zeroinitializer, align 16
} // namespace test3
