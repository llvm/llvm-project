// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -enable-single-byte-coverage=true -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name single-byte-counters.cpp %s | FileCheck %s

// CHECK: testIf
int testIf(int x) { // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+8]]:2 = [[C00:#0]]
  int result = 0;
  if (x == 0)       // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = [[C00]]

                    // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:14 -> [[@LINE+1]]:5 = [[C0T:#1]]
    result = -1;    // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:16 = [[C0T]]

  return result;    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C0E:#2]]
}

// CHECK-NEXT: testIfElse
int testIfElse(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+9]]:2 = [[C10:#0]]
  int result = 0;
  if (x < 0)            // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C10]]

                        // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C1T:#1]]
    result = 0;         // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:15 = [[C1T]]
  else                  // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:16 -> [[@LINE+1]]:5 = [[C1F:#2]]
    result = x * x;     // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:19 = [[C1F]]
  return result;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C1E:#3]]
}

// CHECK-NEXT: testIfElseReturn
int testIfElseReturn(int x) { // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+10]]:2 = [[C20:#0]]
  int result = 0;
  if (x > 0)                  // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C20]]

                              // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C2T:#1]]
    result = x * x;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:19 = [[C2T]]
  else                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:20 -> [[@LINE+1]]:5 = [[C2F:#2]]
    return 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:13 = [[C2F]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:3 = [[C2E:#3]]
  return result;              // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C2E:#3]]
}

// CHECK-NEXT: testIfBothReturn
int testIfBothReturn(int x) { // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+10]]:2 = [[C20:#0]]
  int result = 0;
  if (x > 0)                  // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C20]]

                              // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C2T:#1]]
    return 42;                // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:14 = [[C2T]]
  else                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:15 -> [[@LINE+1]]:5 = [[C2F:#2]]
    return 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:13 = [[C2F]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:3 = #3
  return -1;                  // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:12 = #3
}

// CHECK-NEXT: testSwitch
int testSwitch(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+20]]:2 = [[C30:#0]]
  int result;
  switch (x) {
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+13]]:15 = 0
  case 1:               // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:10 = [[C31:#2]]

    result = 1;
    break;
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = 0
  case 2:               // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:10 = [[C32:#3]]

    result = 2;
    break;
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = 0
  default:              // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:15 = [[C3D:#4]]

    result = 0;
  }
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE+1]]:3 = [[C3E:#1]]
  return result;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C3E]]
}

// CHECK-NEXT: testWhile
int testWhile() {       // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+12]]:2 = [[C40:#0]]
  int i = 0;
  int sum = 0;
  while (i < 10) {      // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:16 = [[C4C:#1]]

                        // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:17 -> [[@LINE-2]]:18 = [[C4T:#2]]
                        // CHECK-NEXT: File 0, [[@LINE-3]]:18 -> [[@LINE+3]]:4 = [[C4T]]
    sum += i;
    i++;
  }

  return sum;           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = [[C4E:#3]]
}

// CHECK-NEXT: testContinueBreak
int testContinueBreak() { // CHECK-NEXT: File 0, [[@LINE]]:25 -> [[@LINE+23]]:2 = #0
  int i = 0;
  int sum = 0;
  while (i < 10) {   // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:16 = #1

                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:17 -> [[@LINE-2]]:18 = [[C5B:#2]]
                     // CHECK-NEXT: File 0, [[@LINE-3]]:18 -> [[@LINE+14]]:4 = [[C5B]]
    if (i == 4)      // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:15 = [[C5B]]

                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:16 -> [[@LINE+1]]:7 = [[C5T:#4]]
      continue;      // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:15 = [[C5T]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:16 -> [[@LINE+2]]:5 = [[C5F:#5]]
                     // CHECK-NEXT: File 0, [[@LINE+1]]:5 -> [[@LINE+8]]:4 = [[C5F]]
    if (i == 5)      // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:15 = [[C5F]]

                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:16 -> [[@LINE+1]]:7 = [[C5T1:#6]]
      break;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C5T1]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:13 -> [[@LINE+1]]:5 = [[C5F1:#7]]
    sum += i;        // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+2]]:4 = [[C5F1]]
    i++;
  }
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE+1]]:3 = [[C5E:#3]]
  return sum;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = [[C5E]]
}

// CHECK-NEXT: testFor
int testFor() { // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE+13]]:2 = [[C60:#0]]
  int i;
  int sum = 0;
                // CHECK-NEXT: File 0, [[@LINE+3]]:19 -> [[@LINE+3]]:25 = [[C61:#1]]

                // CHECK-NEXT: File 0, [[@LINE+1]]:27 -> [[@LINE+1]]:30 = [[C6C:#2]]
  for (int i = 0; i < 10; i++) {
                // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:31 -> [[@LINE-1]]:32 = [[C6B:#3]]
                // CHECK-NEXT: File 0, [[@LINE-2]]:32 -> [[@LINE+2]]:4 = [[C6B]]
    sum += i;
  }

  return sum;   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = [[C6E:#4]]
}

// CHECK-NEXT: testForRange
int testForRange() {    // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+11]]:2 = [[C70:#0]]
  int sum = 0;
  int array[] = {1, 2, 3, 4, 5};

  for (int element : array) {
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:28 -> [[@LINE-1]]:29 = [[C7B:#1]]
                        // CHECK-NEXT: File 0, [[@LINE-2]]:29 -> [[@LINE+2]]:4 = [[C7B]]
      sum += element;
  }

  return sum;           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = [[C7E:#2]]
}

// CHECK-NEXT: testDo
int testDo() {          // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+9]]:2 = [[C80:#0]]
  int i = 0;
  int sum = 0;
  do {                  // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE+3]]:4 = [[C8B:#1]]
    sum += i;
    i++;
  } while (i < 5);      // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE]]:17 = [[C8C:#2]]

  return sum;           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = [[C8E:#3]]
}

// CHECK-NEXT: testConditional
int testConditional(int x) {    // CHECK-NEXT: File 0, [[@LINE]]:28 -> [[@LINE+7]]:2 = [[C90:#0]]
 int result = (x > 0) ? 1 : -1; // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE]]:22 = [[C90]]

                                // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:24 -> [[@LINE-2]]:25 = [[C9T:#2]]
                                // CHECK-NEXT: File 0, [[@LINE-3]]:25 -> [[@LINE-3]]:26 = [[C9T]]
                                // CHECK-NEXT: File 0, [[@LINE-4]]:29 -> [[@LINE-4]]:31 = [[C9F:#3]]
 return result;                 // CHECK-NEXT: File 0, [[@LINE]]:2 -> [[@LINE]]:15 = [[C9E:#1]]
}
