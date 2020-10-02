// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple %itanium_abi_triple -std=c++11 -fexceptions -fcxx-exceptions -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name trymacro.cpp %s | FileCheck %s

// CHECK: Z3fn1v:
void fn1() try { return; } // CHECK: [[@LINE]]:12 -> [[@LINE+1]]:14 = #1
catch(...) {}              // CHECK: [[@LINE]]:12 -> [[@LINE]]:14 = #2

#define RETURN_BLOCK { return; }

// CHECK: Z3fn2v:
void fn2() try RETURN_BLOCK // CHECK: [[@LINE]]:12 -> [[@LINE+1]]:14 = #1
catch(...) {}               // CHECK: [[@LINE]]:12 -> [[@LINE]]:14 = #2

#define TRY try
#define CATCH(x) catch (x)

// CHECK: Z3fn3v:
void fn3() TRY { return; } // CHECK: [[@LINE]]:15 -> [[@LINE+1]]:14 = #1
CATCH(...) {}              // CHECK: [[@LINE]]:12 -> [[@LINE]]:14 = #2

// CHECK: Z3fn4v:
#define TRY2 try { // CHECK-DAG: File 1, [[@LINE]]:18 -> [[@LINE]]:19 = #1
void fn4() TRY2 // CHECK-DAG: Expansion,File 0, [[@LINE]]:12 -> [[@LINE]]:16 = #1 (Expanded file = 1)
  for (;;)
    return;
}
catch (...) {}

// CHECK: Z3fn5v:
#define TRY3 try { return; } catch (...) // CHECK-DAG: File 2, [[@LINE]]:18 -> [[@LINE]]:29 = #1
#define TRY4 try { TRY3 { return; } } catch (...) // CHECK-DAG: Expansion,File 1, [[@LINE]]:20 -> [[@LINE]]:24 = #1 (Expanded file = 2)
void fn5() {
  for (;;) {
    TRY4 { return; } // CHECK-DAG: Expansion,File 0, [[@LINE]]:5 -> [[@LINE]]:9 = #1 (Expanded file = 1)
  }                  // CHECK-DAG: File 0, [[@LINE-1]]:10 -> [[@LINE-1]]:21 = #5
}

int main() {
  fn1();
  fn2();
  fn3();
  fn4();
  fn5();
}
