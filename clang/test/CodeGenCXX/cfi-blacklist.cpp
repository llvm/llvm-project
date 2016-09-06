// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOBL %s
// RUN: echo "type:std::*" > %t.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s

struct S1 {
  virtual void f();
};

namespace std {

struct S2 {
  virtual void f();
};

}

// CHECK: define{{.*}}s1f
// NOBL: llvm.type.test
// NOSTD: llvm.type.test
void s1f(S1 *s1) {
  s1->f();
}

// CHECK: define{{.*}}s2f
// NOBL: llvm.type.test
// NOSTD-NOT: llvm.type.test
void s2f(std::S2 *s2) {
  s2->f();
}
