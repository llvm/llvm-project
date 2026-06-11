// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

inline int outer() {
  static int s = 123;
  struct L {
    __attribute__((used)) static int get() { return s; }
  };
  return s;
}

// CIR: cir.global linkonce_odr comdat @_ZZ5outervE1s = #cir.int<123> : !s32i
// CIR-NOT: @_ZZ5outervE1s.1
// CIR: cir.func{{.*}}@_Z5outerv()
// CIR: cir.get_global @_ZZ5outervE1s
// CIR: cir.func{{.*}}@_ZZ5outervEN1L3getEv()
// CIR: cir.get_global @_ZZ5outervE1s

// LLVM: @_ZZ5outervE1s = linkonce_odr global i32 123, comdat, align 4
// LLVM-NOT: @_ZZ5outervE1s.1
// LLVM-DAG: define {{.*}}@_Z5outerv()
// LLVM-DAG: define {{.*}}@_ZZ5outervEN1L3getEv()
