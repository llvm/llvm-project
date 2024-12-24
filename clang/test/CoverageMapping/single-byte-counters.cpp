// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -enable-single-byte-coverage=true -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name single-byte-counters.cpp %s | FileCheck %s

// CHECK: testIf
int testIf(int x) { // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+8]]:2 = [[C00:#0]]
  int result = 0;
  if (x == 0)       // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = [[C00]]
                    // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:13 = [[C0T:#1]], [[C0F:#2]]
                    // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:14 -> [[@LINE+1]]:5 = [[C0T]]
    result = -1;    // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:16 = [[C0T]]

  return result;    // #0
}

// CHECK-NEXT: testIfElse
int testIfElse(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+9]]:2 = [[C10:#0]]
  int result = 0;
  if (x < 0)            // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C10]]
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:12 = [[C1T:#1]], [[C1F:#2]]
                        // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C1T]]
    result = 0;         // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:15 = [[C1T]]
  else                  // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:16 -> [[@LINE+1]]:5 = [[C1F]]
    result = x * x;     // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:19 = [[C1F]]
  return result;        // #0
}

// CHECK-NEXT: testIfElseReturn
int testIfElseReturn(int x) { // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+10]]:2 = [[C20:#0]]
  int result = 0;
  if (x > 0)                  // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C20]]
                              // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:12 = [[C2T:#1]], [[C2F:#2]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C2T]]
    result = x * x;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:19 = [[C2T]]
  else                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:20 -> [[@LINE+1]]:5 = [[C2F]]
    return 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:13 = [[C2F]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:3 = [[C2T]]
  return result;              // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C2T]]
}

// CHECK-NEXT: testIfBothReturn
int testIfBothReturn(int x) { // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+10]]:2 = [[C20:#0]]
  int result = 0;
  if (x > 0)                  // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C20]]
                              // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:12 = [[C2T:#1]], [[C2F:#2]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = [[C2T]]
    return 42;                // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:14 = [[C2T]]
  else                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:15 -> [[@LINE+1]]:5 = [[C2F]]
    return 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:13 = [[C2F]]
                              // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:3 = 0
  return -1;                  // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:12 = 0
}

// CHECK-NEXT: testSwitch
int testSwitch(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+20]]:2 = [[C30:#0]]
  int result;
  switch (x) {
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+13]]:15 = 0
  case 1:               // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:10 = [[C31:#2]]
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:3 -> [[@LINE-1]]:9 = [[C31]], 0
    result = 1;
    break;
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = 0
  case 2:               // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:10 = [[C32:#3]]
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:3 -> [[@LINE-1]]:9 = [[C32]], 0
    result = 2;
    break;
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:11 -> [[@LINE+1]]:3 = 0
  default:              // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:15 = [[C3D:#4]]
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:3 -> [[@LINE-1]]:10 = [[C3D]], 0
    result = 0;
  }
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE+1]]:3 = [[C3E:#1]]
  return result;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:16 = [[C3E]]
}

// CHECK-NEXT: testWhile
int testWhile() {       // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+12]]:2 = [[C40:#0]]
  int i = 0;
  int sum = 0;
  while (i < 10) {      // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:16 = ([[C40]] + [[C4T:#1]])
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:10 -> [[@LINE-1]]:16 = [[C4T]], [[C4F:#0]]
                        // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:17 -> [[@LINE-2]]:18 = [[C4T]]
                        // CHECK-NEXT: File 0, [[@LINE-3]]:18 -> [[@LINE+3]]:4 = [[C4T]]
    sum += i;
    i++;
  }

  return sum;           // #0
}

// CHECK-NEXT: testContinueBreak
int testContinueBreak() { // CHECK-NEXT: File 0, [[@LINE]]:25 -> [[@LINE+23]]:2 = [[C50:#0]]
  int i = 0;
  int sum = 0;
  while (i < 10) {   // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:16 = (([[C50]] + [[C5T:#2]]) + [[C5F1:#5]])
                     // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:10 -> [[@LINE-1]]:16 = [[C5B:#1]], [[C5E:#6]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:17 -> [[@LINE-2]]:18 = [[C5B]]
                     // CHECK-NEXT: File 0, [[@LINE-3]]:18 -> [[@LINE+14]]:4 = [[C5B]]
    if (i == 4)      // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:15 = [[C5B]]
                     // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:9 -> [[@LINE-1]]:15 = [[C5T]], [[C5F:#4]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:16 -> [[@LINE+1]]:7 = [[C5T]]
      continue;      // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:15 = [[C5T]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:16 -> [[@LINE+2]]:5 = [[C5F]]
                     // CHECK-NEXT: File 0, [[@LINE+1]]:5 -> [[@LINE+8]]:4 = [[C5F]]
    if (i == 5)      // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:15 = [[C5F]]
                     // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:9 -> [[@LINE-1]]:15 = [[C5T1:#3]], [[C5F1]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:16 -> [[@LINE+1]]:7 = [[C5T1]]
      break;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = [[C5T1]]
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:13 -> [[@LINE+1]]:5 = [[C5F1]]
    sum += i;        // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+2]]:4 = [[C5F1]]
    i++;
  }
                     // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE+1]]:3 = ([[C5T1]] + [[C5E]])
  return sum;        // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:13 = ([[C5T1]] + [[C5E]])
}

// CHECK-NEXT: testFor
int testFor() { // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE+13]]:2 = [[C60:#0]]
  int i;
  int sum = 0;
                // CHECK-NEXT: File 0, [[@LINE+3]]:19 -> [[@LINE+3]]:25 = ([[C60]] + [[C6B:#1]])
                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:19 -> [[@LINE+2]]:25 = [[C6B]], [[C6E:#0]]
                // CHECK-NEXT: File 0, [[@LINE+1]]:27 -> [[@LINE+1]]:30 = [[C6B]]
  for (int i = 0; i < 10; i++) {
                // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:31 -> [[@LINE-1]]:32 = [[C6B]]
                // CHECK-NEXT: File 0, [[@LINE-2]]:32 -> [[@LINE+2]]:4 = [[C6B]]
    sum += i;
  }

  return sum;   // #0
}

// CHECK-NEXT: testForRange
int testForRange() {    // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+12]]:2 = [[C70:#0]]
  int sum = 0;
  int array[] = {1, 2, 3, 4, 5};

                        // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:20 -> [[@LINE+1]]:21 = [[C7B:#1]], [[C7E:#0]]
  for (int element : array) {
                        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:28 -> [[@LINE-1]]:29 = [[C7B]]
                        // CHECK-NEXT: File 0, [[@LINE-2]]:29 -> [[@LINE+2]]:4 = [[C7B]]
      sum += element;
  }

  return sum;           // #0
}

// CHECK-NEXT: testDo
int testDo() {          // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+10]]:2 = [[C80:#0]]
  int i = 0;
  int sum = 0;
  do {                  // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE+3]]:4 = ([[C80]] + [[C8B:#1]])
    sum += i;
    i++;
  } while (i < 5);      // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE]]:17 = ([[C80]] + [[C8B]])
                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:12 -> [[@LINE-1]]:17 = [[C8B]], [[C8E:#0]]

  return sum;           // #0
}

// CHECK-NEXT: testConditional
int testConditional(int x) {    // CHECK-NEXT: File 0, [[@LINE]]:28 -> [[@LINE+7]]:2 = [[C90:#0]]
 int result = (x > 0) ? 1 : -1; // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE]]:22 = [[C90]]
                                // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:22 = [[C9T:#1]], [[C9F:#2]]
                                // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:24 -> [[@LINE-2]]:25 = [[C9T]]
                                // CHECK-NEXT: File 0, [[@LINE-3]]:25 -> [[@LINE-3]]:26 = [[C9T]]
                                // CHECK-NEXT: File 0, [[@LINE-4]]:29 -> [[@LINE-4]]:31 = [[C9F]]
 return result;                 // #0
}
