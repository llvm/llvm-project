// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

// CHECK-DAG: @_ZZ2f0vE1a
// CHECK-DAG: @_ZZ2f0vE1c
// CHECK-DAG: @_ZZ2f0vE1e_0
inline int f0() {
  static union {
    int a;
    long int b;
  };

  static union {
    int c;
    double d;
  };

  if (0) {
    static union {
      int e;
      int f;
    };
  }
  static union {
    int e;
    int f;
  };

  return a+c;
}

inline void nop() {
  static union {
    union {
    };
  };
}

int f1 (int a, int c) {
  nop();
  return a+c+f0();
}

