// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fexceptions -fcxx-exceptions -emit-llvm-only -triple %itanium_abi_triple -main-file-name terminate-statements.cpp -I %S/Inputs %s | FileCheck %s

int f1() {
  return 0;
  return 0; // CHECK: Gap,File 0, [[@LINE-1]]:12 -> [[@LINE]]:3 = 0
}

int f2(int i) {
  if (i)
    return 0;
  else
    ;       // CHECK: Gap,File 0, [[@LINE]]:6 -> [[@LINE+1]]:3 = (#0 - #1)
  return 1; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f3() {
  for (int a = 1; a < 9; a--)
    return a; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = (#0 - #1)
  return 0;   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f4(int i) {
  while (i > 0) {
    i++;
    return i;
  }         // CHECK: File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #1)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f5(int i) {
  do {
    return i;
  } while (i > 0); // CHECK: Gap,File 0, [[@LINE]]:19 -> [[@LINE+1]]:3 = (0 - #1)
  return 0;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (0 - #1)
}

int f6() {
  int arr[] = {1, 2, 3, 4};
  for (int i : arr) {
    return i;
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #1)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f7() {
  {
    {
      return 0;
    }
    return 0; // CHECK: Gap,File 0, [[@LINE-1]]:6 -> [[@LINE]]:5 = 0
  }
  return 0; // CHECK: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE]]:3 = 0
}

int f8(int i) {
  if (i == 1)
    return 1; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = (#0 - #1)
  if (i == 2) // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+5]]:2 = (#0 - #1)
    return 2; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = ((#0 - #1) - #2)
  if (i == 3) // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:2 = ((#0 - #1) - #2)
    return 3; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = (((#0 - #1) - #2) - #3)
  return 4;   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (((#0 - #1) - #2) - #3)
}

int f9(int i) {
  if (i == 1)
    return 1;      // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:8 = (#0 - #1)
  else if (i == 2) // CHECK-NEXT: File 0, [[@LINE]]:8 -> [[@LINE+1]]:13 = (#0 - #1)
    return 2;      // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = ((#0 - #1) - #2)
  return 3;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = ((#0 - #1) - #2)
}

int f10(int i) {
  if (i == 1) {
    return 0;
    if (i == 2) // CHECK: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE]]:5 = 0
      return 0;
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = ((#0 - #1) - #2)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = ((#0 - #1) - #2)
}

int f11(int i) {
  if (i == 1)
    i = 2;
  else
    return 0; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:3 = #1
  return 0;   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = #1
}

int f12(int i) {
  int x = 1;
  if (x == 1) {
    if (x == 1) {
      return 0;
    }
  } else if (x == 2) {
    x = 2;
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #2)
  return 1; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #2)
}

int f13(int i) {
  if (i == 1) {
    return 0;     // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:5 = 0
    if (i == 2) { // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+3]]:4 = 0
      i++;
    }
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0  - #1)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f14(int i) {
  while (i == 0) {
    while (i < 10) {
      i++;
      return 0;
    }
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #2)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #2)
}

int f15(int i) {
  while (i == 0) {
    return 0;        // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:5 = 0
    while (i < 10) { // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+3]]:4 = 0
      i++;
    }
  }         // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #1)
  return 0; // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = (#0 - #1)
}

int f16(int i) {
  while (i == 0) {
    break;
    return 0;
  }
  return 0; // CHECK-NOT: Gap,File 0, [[@LINE-1]]
}

#define IF if
#define STMT(S) S

// CHECK-LABEL: _Z3fooi:
void foo(int x) {
  if (x == 0) {
    return;
  } // CHECK-NOT: Gap,File 0, [[@LINE]]:4
    //< Don't complete the last deferred region in a decl, even though it may
    //< leave some whitespace marked with the same counter as the final return.
}

// CHECK-LABEL: _Z4foooi:
void fooo(int x) {
  if (x == 0) {
    return;
  } // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+2]]:3 = (#0 - #1)

  if (x == 1) {
    return;
  } // CHECK-NOT: Gap,File 0, [[@LINE]]:4

}

// CHECK-LABEL: _Z3bazv:
void baz() { // CHECK: [[@LINE]]:12 -> [[@LINE+2]]:2
  return;    // CHECK-NOT: File
}

// CHECK-LABEL: _Z4maazv:
void maaz() {
  if (true)
    return; // CHECK: Gap,File 0, [[@LINE]]:12
  else
    return; // CHECK-NOT: Gap,File 0, [[@LINE]]
}

// CHECK-LABEL: _Z5maaazv:
void maaaz() {
  if (true) {
    return;
  } else {  // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE]]:10
    return; // CHECK-NOT: Gap,File 0, [[@LINE]]
  }
}

// CHECK-LABEL: _Z3bari:
void bar(int x) {
  IF (x)
    return; // CHECK: Gap,File 0, [[@LINE]]:12 -> [[@LINE+2]]:3 = (#0 - #1)

  IF (!x)
    return; // CHECK: Gap,File 0, [[@LINE]]:12 -> [[@LINE+2]]:3 = ((#0 - #1) - #2)

  foo(x);
}

// CHECK-LABEL: _Z4quuxi:
void quux(int x) {
  STMT(
  if (x == 0)
    return;)

  // CHECK: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+2]]:3 = (#0 - #1)

  if (x == 1)
    STMT(return;)

  // CHECK: Gap,File 0, [[@LINE-2]]:18 -> [[@LINE+2]]:3 = ((#0 - #1) - #2)

  STMT(
  if (x == 2)
    return;

  // CHECK-NOT: [[@LINE-2]]:{{.*}} -> [[@LINE+2]]

  if (x == 3)
    return;
  )
}

// CHECK-LABEL: _Z8weird_ifv:
void weird_if() {
  int i = 0;

  if (false)
    return; // CHECK: Gap,File 0, [[@LINE]]:12 -> [[@LINE+2]]:3 = (#0 - #1)

  if (false)
    i++;

  if (i + 100 > 0) { // CHECK: [[@LINE]]:20 -> [[@LINE+6]]:4 = #3
    if (false)       // CHECK: [[@LINE+1]]:7 -> [[@LINE+1]]:13 = #4
      return;        // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+2]]:5 = (#3 - #4)
                     // CHECK: [[@LINE+1]]:5 -> [[@LINE+1]]:11 = (#3 - #4)
    return;

  }                  // CHECK: Gap,File 0, [[@LINE]]:4 -> [[@LINE+2]]:3 = ((#0 - #1) - #3)

  if (false)
    return; // CHECK-NOT: Gap,File 0, [[@LINE]]:11
}

// CHECK-LABEL: _Z8for_loopv:
void for_loop() {
  if (false)
    return; // CHECK: Gap,File 0, [[@LINE]]:12 -> [[@LINE+2]]:3 = (#0 - #1)

  for (int i = 0; i < 10; ++i) {
    if (i % 2 == 0)
      continue; // CHECK: Gap,File 0, [[@LINE]]:16 -> [[@LINE+2]]:5 = (#2 - #3)

    if (i % 5 == 0)
      break; // CHECK: Gap,File 0, [[@LINE]]:13 -> [[@LINE+2]]:5 = ((#2 - #3) - #4)

    int x = i; // CHECK: [[@LINE]]:5 -> [[@LINE+1]]:11 = ((#2 - #3) - #4)
    return; // CHECK-NOT: [[@LINE]]:11 -> [[@LINE+2]]

  }
}

struct Error {};

// CHECK-LABEL: _Z10while_loopv:
void while_loop() {
  if (false)
    return; // CHECK: Gap,File 0, [[@LINE]]:12 -> [[@LINE+2]]:3 = (#0 - #1)

  int x = 0;
  while (++x < 10) {
    if (x == 1)
      continue; // CHECK: Gap,File 0, [[@LINE]]:16 -> [[@LINE+2]]:5 = (#2 - #3)

    while (++x < 4) {
      if (x == 3)
        break; // CHECK: Gap,File 0, [[@LINE]]:15 -> [[@LINE+2]]:7 = (#4 - #5)

      while (++x < 5) {}
    }

    if (x == 0)
      throw Error(); // CHECK: Gap,File 0, [[@LINE]]:21 -> [[@LINE+2]]:5 = ((#2 - #3) - #7)

    while (++x < 9) {
      if (x == 0)
        break; // CHECK-NOT: [[@LINE]]:14 -> [[@LINE+2]]

    }
  }
}

// CHECK-LABEL: _Z5gotosv:
void gotos() {
  if (false)
    goto out; // CHECK: Gap,File 0, [[@LINE]]:14 -> [[@LINE+2]]:3 = (#0 - #1)

  return; // CHECK: [[@LINE]]:3 -> [[@LINE]]:9 = (#0 - #1)

out:
	return; // CHECK-NOT: Gap,File 0, [[@LINE]]:8
}

// CHECK-LABEL: _Z8switchesv:
void switches() {
  int x;
  switch (x) {
    case 0:
      return;
    default:
      return; // CHECK-NOT: Gap,File 0, [[@LINE]]
  }
}

#include "deferred-region-helper.h"
// CHECK-LABEL: _Z13included_funcv:
// CHECK:  Gap,File 0, 2:13 -> 3:5 = #1
// CHECK:  Gap,File 0, 3:12 -> 4:3 = (#0 - #1)

// CHECK-LABEL: _Z7includev:
void include() {
  included_func();
}

int main() {
  foo(0);
  foo(1);
  fooo(0);
  fooo(1);
  maaz();
  maaaz();
  baz();
  bar(0);
  bar(1);
  quux(0);
  quux(1);
  quux(2);
  quux(3);
  weird_if();
  for_loop();
  while_loop();
  gotos();
  include();
  return 0;
}
