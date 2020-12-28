// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name label.cpp %s | FileCheck %s

// CHECK: func
void func() {                // CHECK-NEXT: File 0, [[@LINE]]:13 -> {{[0-9]+}}:2 = #0
  int i = 0;                 // CHECK-NEXT: File 0, [[@LINE+3]]:14 -> [[@LINE+3]]:20 = (#0 + #3)
                             // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:14 -> [[@LINE+2]]:20 = #1, ((#0 + #3) - #1)
                             // CHECK-NEXT: File 0, [[@LINE+1]]:22 -> [[@LINE+1]]:25 = #3
  for(i = 0; i < 10; ++i) {  // CHECK: File 0, [[@LINE]]:27 -> [[@LINE+11]]:4 = #1
                             // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:13 = #1
    if(i < 5) {              // CHECK: File 0, [[@LINE]]:15 -> [[@LINE+6]]:6 = #2
      {
        x:                   // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE+4]]:6 = #3
          int j = 1;
      }
      int m = 2;
    } else
      goto x;                // CHECK: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = (#1 - #2)
    int k = 3;               // CHECK-NEXT: File 0, [[@LINE-1]]:13 -> [[@LINE]]:5 = #3
  }                          // CHECK-NEXT: File 0, [[@LINE-1]]:5 -> [[@LINE]]:4 = #3
  static int j = 0;          // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+4]]:2 = ((#0 + #3) - #1)
  ++j;
  if(j == 1)                 // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:12 = ((#0 + #3) - #1)
    goto x;                  // CHECK: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #4
}

                             // CHECK-NEXT: test1
void test1(int x) {          // CHECK-NEXT: File 0, [[@LINE]]:19 -> {{[0-9]+}}:2 = #0
  if(x == 0)                 // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:12 = #0
    goto a;                  // CHECK: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #1
                             // CHECK-NEXT: File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = (#0 - #1)
  goto b;                    // CHECK: File 0, [[@LINE]]:3 -> [[@LINE+5]]:2 = (#0 - #1)
                             // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:9 -> [[@LINE+1]]:1 = #2
a:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+3]]:2 = #2
b:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+2]]:2 = #3
  x = x + 1;
}

                             // CHECK-NEXT: test2
void test2(int x) {          // CHECK-NEXT: File 0, [[@LINE]]:19 -> {{[0-9]+}}:2 = #0
  if(x == 0)                 // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:12 = #0
    goto a;                  // CHECK: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #1
                             // CHECK: Gap,File 0, [[@LINE-1]]:12 -> [[@LINE+3]]:8 = (#0 - #1)
                             // CHECK-NEXT: File 0, [[@LINE+2]]:8 -> [[@LINE+3]]:11 = (#0 - #1)
                             // CHECK-NEXT: File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:17 = (#0 - #1)
  else if(x == 1)            // CHECK: File 0, [[@LINE+1]]:5 -> [[@LINE+1]]:11 = #2
    goto b;                  // CHECK-NEXT: File 0, [[@LINE]]:11 -> [[@LINE+1]]:1 = #3
a:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+3]]:2 = #3
b:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+2]]:2 = #4
  x = x + 1;
}

// CHECK-NEXT: test3
#define a b
void test3() {
  if (0)
    goto b; // CHECK: Gap,File 0, [[@LINE]]:11 -> [[@LINE+1]]:1 = [[retnCount:#[0-9]+]]
a: // CHECK-NEXT: Expansion,File 0, [[@LINE]]:1 -> [[@LINE]]:2 = [[retnCount]] (Expanded file = 1)
  return; // CHECK-NEXT: File 0, [[@LINE-1]]:2 -> [[@LINE]]:9 = [[retnCount]]
}
#undef a

                             // CHECK: main
int main() {                 // CHECK-NEXT: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int j = 0;
  for(int i = 0; i < 10; ++i) { // CHECK: File 0, [[@LINE]]:31 -> [[@LINE+13]]:4 = #1
  a:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+12]]:4 = #2
    if(i < 3)                // CHECK-NEXT: File 0, [[@LINE]]:8 -> [[@LINE]]:13 = #2
      goto e;                // CHECK: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = #3
                             // CHECK-NEXT: File 0, [[@LINE-1]]:13 -> [[@LINE+1]]:5 = (#2 - #3)
    goto c;                  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+8]]:4 = (#2 - #3)
                             // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = #4
  b:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+6]]:4 = #4
    j = 2;
  c:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+4]]:4 = #5
    j = 1;
                             // CHECK-NEXT: File 0, [[@LINE+1]]:3 -> [[@LINE+2]]:4 = #6
  e: f: ;                    // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE+1]]:4 = #7
  }
  func();                    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:2 = ((#0 + #7) - #1)
  test1(0);
  test2(2);
}
