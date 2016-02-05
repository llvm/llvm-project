// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name test.c %s | FileCheck %s

void bar();
static void static_func();

                                 // CHECK: main
int main() {                     // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+7]]:2 = #0
                                 // CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:24 = (#0 + #1)
  for(int i = 0; i < 10; ++i) {  // CHECK-NEXT: File 0, [[@LINE]]:26 -> [[@LINE]]:29 = #1
    bar();                       // CHECK-NEXT: File 0, [[@LINE-1]]:31 -> [[@LINE+1]]:4 = #1
  }
  static_func();
  return 0;
}

                                 // CHECK-NEXT: foo
void foo() {                     // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+5]]:2 = #0
                                 // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:7 = #0
  if(1) {                        // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE+2]]:4 = #1
    int i = 0;
  }
}

                                 // CHECK-NEXT: bar
void bar() {                     // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+1]]:2 = #0
}

                                 // CHECK-NEXT: static_func
void static_func() { }           // CHECK: File 0, [[@LINE]]:20 -> [[@LINE]]:23 = #0

                                 // CHECK-NEXT: func
static void func() { }           // CHECK: File 0, [[@LINE]]:20 -> [[@LINE]]:23 = 0
