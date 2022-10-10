// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++2b -triple %itanium_abi_triple -main-file-name if.cpp %s | FileCheck %s

int nop() { return 0; }

// CHECK-LABEL: _Z3foov:
                                // CHECK-NEXT: [[@LINE+3]]:12 -> [[@LINE+8]]:2 = #0
                                // CHECK-NEXT: [[@LINE+3]]:15 -> [[@LINE+3]]:19 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:15 -> [[@LINE+2]]:19 = 0, 0
void foo() {                    // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:21 -> [[@LINE+1]]:22 = #2
  if (int j = true ? nop()      // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = #2
                   : nop();     // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = (#0 - #2)
      j)                        // CHECK-NEXT: [[@LINE]]:7 -> [[@LINE]]:8 = #0
    ++j;                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:8 = #1, (#0 - #1)
}                               // CHECK-NEXT: [[@LINE-2]]:9 -> [[@LINE-1]]:5 = #1
                                // CHECK-NEXT: [[@LINE-2]]:5 -> [[@LINE-2]]:8 = #1

// FIXME: Do not generate coverage for discarded branches in if consteval and if constexpr statements
constexpr int check_consteval(int i) {
    if consteval {
      i++;
    }
    if !consteval {
      i++;
    }
    if consteval {
        return 42;
    } else {
        return i;
    }
}

// CHECK-LABEL: main:
int main() {                    // CHECK: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int i = 0;
                                // CHECK-NEXT: File 0, [[@LINE+3]]:6 -> [[@LINE+3]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #1, (#0 - #1)
                                // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:14 = #1
  if(i == 0) i = 1;             // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE]]:19 = #1

                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #2, (#0 - #2)
  if(i == 1)                    // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:5 = #2
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = #2

                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #3, (#0 - #3)
  if(i == 0) { i = 1;           // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #3
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #3
  }
                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #4, (#0 - #4)
  if(i != 0) {                  // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #4
    i = 1;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #4
  } else {                      // CHECK-NEXT: Gap,File 0, [[@LINE]]:4 -> [[@LINE]]:10 = (#0 - #4)
    i = 3;                      // CHECK-NEXT: File 0, [[@LINE-1]]:10 -> [[@LINE+1]]:4 = (#0 - #4)
  }

                                // CHECK-NEXT: File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:13 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:13 = #5, (#0 - #5)
  i = i == 0?                   // CHECK-NEXT: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:9 = #5
        i + 1 :                 // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = #5
        i + 2;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = (#0 - #5)

                                // CHECK-NEXT: File 0, [[@LINE+3]]:7 -> [[@LINE+3]]:13 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:13 = #6, (#0 - #6)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:14 -> [[@LINE+1]]:20 = #6
  i = i == 0?i + 12:i + 10;     // CHECK-NEXT: File 0, [[@LINE]]:21 -> [[@LINE]]:27 = (#0 - #6)

  // GH-57377
  constexpr int c_i = check_consteval(0);
  check_consteval(i);

  return 0;
}

#define FOO true

// CHECK-LABEL: _Z7ternaryv:
void ternary() {
  true ? FOO : FOO; // CHECK-NOT: Gap,{{.*}}, [[@LINE]]:8 ->
}
