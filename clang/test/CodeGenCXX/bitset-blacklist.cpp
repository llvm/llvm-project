// RUN: echo "type:attr:uuid" > %t.txt
// RUN: %clang_cc1 -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOUUID %s
// RUN: %clang_cc1 -fms-extensions -fwhole-program-vtables -fwhole-program-vtables-blacklist=%t.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOUUID %s
// RUN: echo "type:std::*" > %t.txt
// RUN: %clang_cc1 -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s
// RUN: %clang_cc1 -fms-extensions -fwhole-program-vtables -fwhole-program-vtables-blacklist=%t.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s

struct __declspec(uuid("00000000-0000-0000-0000-000000000000")) S1 {
  virtual void f();
};

namespace std {

struct S2 {
  virtual void f();
};

}

// CHECK: define{{.*}}s1f
// NOSTD: llvm.bitset.test
// NOUUID-NOT: llvm.bitset.test
void s1f(S1 *s1) {
  s1->f();
}

// CHECK: define{{.*}}s2f
// NOSTD-NOT: llvm.bitset.test
// NOUUID: llvm.bitset.test
void s2f(std::S2 *s2) {
  s2->f();
}
