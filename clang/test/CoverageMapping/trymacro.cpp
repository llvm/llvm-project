// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fexceptions -fcxx-exceptions -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name trymacro.cpp %s | FileCheck %s

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

int main() {
  fn1();
  fn2();
  fn3();
}
