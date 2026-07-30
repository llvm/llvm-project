// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name nestedclass.cpp %s > %tmapping
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-OUTER
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-INNER
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-INNERMOST

struct Test {                   // CHECK-OUTER: emitTest
  void emitTest() {             // CHECK-OUTER: File 0, [[@LINE]]:19 -> [[@LINE+2]]:4 = #0
    int i = 0;
  }
  struct Test2 {                // CHECK-INNER: emitTest2
    void emitTest2() {          // CHECK-INNER: File 0, [[@LINE]]:22 -> [[@LINE+2]]:6 = #0
      int i = 0;
    }
    struct Test3 {              // CHECK-INNERMOST: emitTest3
      static void emitTest3() { // CHECK-INNERMOST: File 0, [[@LINE]]:31 -> [[@LINE+2]]:8 = 0
        int i = 0;
      }
    };
  };
};

int main() {
  Test t;
  Test::Test2 t2;
  t.emitTest();
  t2.emitTest2();
  return 0;
}
