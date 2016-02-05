// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name preprocessor.c %s | FileCheck %s

                 // CHECK: func
void func() {    // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+5]]:2 = #0
  int i = 0;
#ifdef MACRO     // CHECK-NEXT: Skipped,File 0, [[@LINE]]:2 -> [[@LINE+2]]:2 = 0
  int x = i;
#endif
}

#if 0
  int g = 0;

  void bar() { }
#endif

                 // CHECK: main
int main() {     // CHECK-NEXT: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int i = 0;
#if 0            // CHECK-NEXT: Skipped,File 0, [[@LINE]]:2 -> [[@LINE+4]]:2 = 0
  if(i == 0) {
    i = 1;
  }
#endif

#if 1
                 // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #0
  if(i == 0) {   // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+2]]:4 = #1
    i = 1;
  }
#else            // CHECK-NEXT: Skipped,File 0, [[@LINE]]:2 -> [[@LINE+5]]:2 = 0
  if(i == 1) {
    i = 0;
  }
}
#endif
  return 0;
}
