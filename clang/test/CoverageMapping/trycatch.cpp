// RUN: %strip_comments > %t.stripped.cpp
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fexceptions -fcxx-exceptions -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name trycatch.cpp %t.stripped.cpp | FileCheck %s

class Error {
};

class ImportantError {
};

class Warning {
};

                                      // CHECK: func
void func(int i) {                    // CHECK-NEXT: File 0, [[@LINE]]:18 -> {{[0-9]+}}:2 = #0
                                      // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:11 = #0
  if(i % 2) {                         // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+4]]:4 = #1
    throw Error();
    int j = 0;                        // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+2]]:4 = 0
                                      // CHECK: File 0, [[@LINE+1]]:10 -> [[@LINE+2]]:27 = (#0 - #1)
  } else if(i == 8)                   // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE]]:19 = (#0 - #1)
    throw ImportantError();           // CHECK: File 0, [[@LINE]]:5 -> [[@LINE]]:27 = #2
}

                                      // CHECK-NEXT: main
int main() {                          // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+13]]:2 = #0
  int j = 1;
  try {                               // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+2]]:4 = #0
    func(j);
  } catch(const Error &e) {           // CHECK-NEXT: File 0, [[@LINE]]:27 -> [[@LINE+2]]:4 = #2
    j = 1;
  } catch(const ImportantError &e) {  // CHECK-NEXT: File 0, [[@LINE]]:36 -> [[@LINE+2]]:4 = #3
    j = 11;
  }
  catch(const Warning &w) {           // CHECK-NEXT: File 0, [[@LINE]]:27 -> [[@LINE+2]]:4 = #4
    j = 0;
  }
  return 0;                           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = #1
}
