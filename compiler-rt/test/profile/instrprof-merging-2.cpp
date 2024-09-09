// UNSUPPORTED: target={{.*windows.*}}

// clang-format off
// RUN: split-file %s %t
// RUN: %clangxx_profgen -fcoverage-mapping %t/test1.cpp -o %t/test1.exe
// RUN: %clangxx_profgen -fcoverage-mapping %t/test2.cpp -o %t/test2.exe
// RUN: env LLVM_PROFILE_FILE=%t/test1.profraw %run %t/test1.exe
// RUN: env LLVM_PROFILE_FILE=%t/test2.profraw %run %t/test2.exe
// RUN: llvm-profdata merge %t/test1.profraw %t/test2.profraw -o %t/merged.profdata
// RUN: llvm-cov show -instr-profile=%t/merged.profdata -object %t/test1.exe %t/test2.exe | FileCheck %s
// RUN: llvm-cov show -instr-profile=%t/merged.profdata -object %t/test2.exe %t/test1.exe | FileCheck %s

// CHECK:       |struct Test {
// CHECK-NEXT: 1|  int getToTest() {
// CHECK-NEXT: 2|    for (int i = 0; i < 1; i++) {
// CHECK-NEXT: 1|      if (false) {
// CHECK-NEXT: 0|        return 1;
// CHECK-NEXT: 0|      }
// CHECK-NEXT: 1|    }
// CHECK-NEXT: 1|    if (true) {
// CHECK-NEXT: 1|      return 1;
// CHECK-NEXT: 1|    }
// CHECK-NEXT: 0|    return 1;
// CHECK-NEXT: 1|  }
// CHECK-NEXT:  |};
// CHECK-NEXT:  |

#--- test.h
struct Test {
  int getToTest() {
    for (int i = 0; i < 1; i++) {
      if (false) {
        return 1;
      }
    }
    if (true) {
      return 1;
    }
    return 1;
  }
};

#--- test1.cpp
#include "test.h"
int main() {
  Test t;
  t.getToTest();
  return 0;
}

#--- test2.cpp
#include "test.h"
int main() {
  return 0;
}
