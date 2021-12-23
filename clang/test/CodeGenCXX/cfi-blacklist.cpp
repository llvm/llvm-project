// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOBL %s

// Check that blacklisting cfi and cfi-vcall work correctly
// RUN: echo "[cfi-vcall]" > %t.vcall.txt
// RUN: echo "type:std::*" >> %t.vcall.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.vcall.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s
//
// RUN: echo "[cfi]" > %t.cfi.txt
// RUN: echo "type:std::*" >> %t.cfi.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.cfi.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s

// Check that blacklisting non-vcall modes does not affect vcalls
// RUN: echo "[cfi-icall|cfi-nvcall|cfi-cast-strict|cfi-derived-cast|cfi-unrelated-cast]" > %t.other.txt
// RUN: echo "type:std::*" >> %t.other.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-blacklist=%t.other.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOBL %s

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
