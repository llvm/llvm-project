// Check that we correctly set or did not set the attribute for each function.
// RUN: %clang_cc1 -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK2

// Check that we correctly set or did not set the attribute on each function despite the
// -mspeculative-load-hardening flag.
// RUN: %clang_cc1 -mspeculative-load-hardening -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK3
// RUN: %clang_cc1 -mspeculative-load-hardening -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK4


// Check that we correctly set or did not set the attribute on each function despite the
// -mno-speculative-load-hardening flag.
// RUN: %clang -mno-speculative-load-hardening -S -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK5
// RUN: %clang -mno-speculative-load-hardening -S -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK6


[[clang::speculative_load_hardening]]
int test1() {
  return 42;
}

int __attribute__((speculative_load_hardening)) test2() {
  return 42;
}

[[clang::no_speculative_load_hardening]]
int test3() {
  return 42;
}

int __attribute__((no_speculative_load_hardening)) test4() {
  return 42;
}
// CHECK1: @{{.*}}test1{{.*}}[[SLH:#[0-9]+]]
// CHECK1: @{{.*}}test3{{.*}}[[NOSLH:#[0-9]+]]
// CHECK1: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK1-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK2: @{{.*}}test2{{.*}}[[SLH:#[0-9]+]]
// CHECK2: @{{.*}}test4{{.*}}[[NOSLH:#[0-9]+]]
// CHECK2: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK2-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK3: @{{.*}}test1{{.*}}[[SLH:#[0-9]+]]
// CHECK3: @{{.*}}test3{{.*}}[[NOSLH:#[0-9]+]]
// CHECK3: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK3-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK4: @{{.*}}test2{{.*}}[[SLH:#[0-9]+]]
// CHECK4: @{{.*}}test4{{.*}}[[NOSLH:#[0-9]+]]
// CHECK4: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK4-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK5: @{{.*}}test1{{.*}}[[SLH:#[0-9]+]]
// CHECK5: @{{.*}}test3{{.*}}[[NOSLH:#[0-9]+]]
// CHECK5: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK5-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK6: @{{.*}}test2{{.*}}[[SLH:#[0-9]+]]
// CHECK6: @{{.*}}test4{{.*}}[[NOSLH:#[0-9]+]]
// CHECK6: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
// CHECK6-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }
