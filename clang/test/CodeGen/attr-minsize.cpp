// RUN: %clang_cc1 -Oz -emit-llvm %s -o - | FileCheck %s -check-prefix=Oz
// RUN: %clang_cc1 -O0 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O1 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O3 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -Os -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// Check that we set the minsize attribute on each function
// when Oz optimization level is set.

int test1() {
  return 42;
// Oz: @{{.*}}test1{{.*}}minsize
// Oz: ret
// OTHER: @{{.*}}test1
// OTHER-NOT: minsize
// OTHER: ret
}

int test2() {
  return 42;
// Oz: @{{.*}}test2{{.*}}minsize
// Oz: ret
// OTHER: @{{.*}}test2
// OTHER-NOT: minsize
// OTHER: ret
}

__attribute__((minsize))
int test3() {
  return 42;
// Oz: @{{.*}}test3{{.*}}minsize
// OTHER: @{{.*}}test3{{.*}}minsize
}

// Check that the minsize attribute is well propagated through
// template instantiation

template<typename T>
__attribute__((minsize))
void test4(T arg) {
  return;
}

template
void test4<int>(int arg);
// Oz: define{{.*}}void @{{.*}}test4
// Oz: minsize
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: minsize

template
void test4<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test4
// Oz: minsize
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: minsize

template<typename T>
void test5(T arg) {
  return;
}

template
void test5<int>(int arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: minsize
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: minsize

template
void test5<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: minsize
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: minsize
