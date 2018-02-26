// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name switchmacro.c %s | FileCheck %s

#define FOO(x) (void)x

// CHECK: foo
int foo(int i) { // CHECK-NEXT: File 0, [[@LINE]]:16 -> {{[0-9]+}}:2 = #0
  switch (i) {
  default:       // CHECK-NEXT: File 0, [[@LINE]]:3 -> {{[0-9]+}}:11 = #2
    if (i == 1)  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:15 = #2
      return 0;  // CHECK: File 0, [[@LINE]]:7 -> [[@LINE]]:15 = #3
                 // CHECK-NEXT: File 0, [[@LINE-1]]:15 -> [[@LINE+3]]:5 = (#2 - #3)
    // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:5 -> [[@LINE+2]]:8 = (#2 - #3) (Expanded file = 1)
    // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> {{[0-9]+}}:11 = (#2 - #3)
    FOO(1);
  case 0:        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:13 = ((#2 + #4) - #3)
    return 2;    // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+6]]:3 = #5

  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:6 = 0
  // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> {{[0-9]+}}:11 = 0
  FOO(1);
  // CHECK-NEXT: File 0, [[@LINE+1]]:3 -> {{[0-9]+}}:11 = #5
  label: ;
  }
}

// PR26825 - Crash when exiting macro expansion containing a switch
// CHECK: bar
#define START { while (0) { switch (0) {
#define END   }}}
void bar() {
  START      // CHECK: File 0, [[@LINE]]:8 -> [[@LINE+2]]:6
default: ;
  END
}

// PR27948 - Crash when handling a switch partially covered by a macro
// CHECK: baz
#define START2 switch (0) default:
void baz() {
  for (;;)
    START2 return; // CHECK: Expansion,File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #1 (Expanded file = 1)
}

int main(int argc, const char *argv[]) {
  foo(3);
  return 0;
}
