// RUN: %clang_cc1 -Oz -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=Oz
// RUN: %clang_cc1     -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O1 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O2 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O3 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -Os -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// Check that we set the minsize attribute on each function
// when Oz optimization level is set.

__attribute__((minsize))
int test1() {
  return 42;
// Oz: @{{.*}}test1{{.*}}[[MINSIZE:#[0-9]+]]
// OTHER: @{{.*}}test1{{.*}}[[MS:#[0-9]+]]
}

int test2() {
  return 42;
// Oz: @{{.*}}test2{{.*}}[[MINSIZE]]
// Oz: ret
// OTHER: @{{.*}}test2
// OTHER-NOT: [[MS]]
// OTHER: ret
}

int test3() {
  return 42;
// Oz: @{{.*}}test3{{.*}}[[MINSIZE]]
// Oz: ret
// OTHER: @{{.*}}test3
// OTHER-NOT: [[MS]]
// OTHER: ret
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
// Oz: [[MINSIZE]]
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: [[MS]]

template
void test4<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test4
// Oz: [[MINSIZE]]
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: [[MS]]

template<typename T>
void test5(T arg) {
  return;
}

template
void test5<int>(int arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: [[MINSIZE]]
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: define{{.*}}void @{{.*}}test5{{.*}}[[MS]]

template
void test5<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: [[MINSIZE]]
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: define{{.*}}void @{{.*}}test5{{.*}}[[MS]]

// Oz: attributes [[MINSIZE]] = { minsize{{.*}} }

// OTHER: attributes [[MS]] = { minsize nounwind{{.*}} }
