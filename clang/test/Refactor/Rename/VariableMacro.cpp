#define Baz Foo // CHECK1-NOT: rename local [[@LINE]]
// CHECK1-NOT: macro [[@LINE-1]]

void foo(int value) {}

void macro() {
  int Foo;  // CHECK1: rename local [[@LINE]]:7 -> [[@LINE]]:10
  Foo = 42; // CHECK1-NEXT: rename local [[@LINE]]:3 -> [[@LINE]]:6
  Baz -= 0; // CHECK1-NEXT: macro [[@LINE]]:3 -> [[@LINE]]:3
  foo(Foo); // CHECK1-NEXT: rename local [[@LINE]]:7 -> [[@LINE]]:10
  foo(Baz); // CHECK1-NEXT: macro [[@LINE]]:7 -> [[@LINE]]:7
}

// RUN: clang-refactor-test rename-initiate -at=%s:7:7 -at=%s:8:3 -at=%s:10:7 -new-name=Bar %s | FileCheck --check-prefix=CHECK1 %s

// RUN: not clang-refactor-test rename-initiate -at=%s:1:13 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:9:3 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// CHECK-ERROR: could not rename symbol at the given location

#define M var
#define MM M
void macro2() {
  int M = 2; // CHECK2: macro [[@LINE]]:7 -> [[@LINE]]:7
  (void)var; // CHECK2-NEXT: rename local [[@LINE]]:9 -> [[@LINE]]:12
  (void)M;   // CHECK2-NEXT: macro [[@LINE]]:9 -> [[@LINE]]:9
  (void)MM;  // CHECK2-NEXT: macro [[@LINE]]:9 -> [[@LINE]]:9
}

// RUN: clang-refactor-test rename-initiate -at=%s:24:9 -new-name=Bar %s | FileCheck --check-prefix=CHECK2 %s

// RUN: not clang-refactor-test rename-initiate -at=%s:20:11 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:21:12 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:23:7 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:25:9 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

#define BAR(x) x
#define FOO(x) BAR(x)
int FOO(global) = 2; // CHECK3: rename [[@LINE]]:9 -> [[@LINE]]:15
void macro3() {
  (void)global;      // CHECK3-NEXT: rename [[@LINE]]:9 -> [[@LINE]]:15
  BAR(global) = 0;   // CHECK3-NEXT: rename [[@LINE]]:7 -> [[@LINE]]:13
}

// RUN: clang-refactor-test rename-initiate -at=%s:40:9 -at=%s:41:7 -new-name=Bar %s | FileCheck --check-prefix=CHECK3 %s



#define CONCAT(x, y) x##_##y
int CONCAT(a, b) = 2; // CHECK4: macro [[@LINE]]:5 -> [[@LINE]]:5
void macro3() {
  (void)a_b;          // CHECK4-NEXT: rename [[@LINE]]:9 -> [[@LINE]]:12
  CONCAT(a, b) = 0;   // CHECK4-NEXT: macro [[@LINE]]:3 -> [[@LINE]]:3
}

// RUN: clang-refactor-test rename-initiate -at=%s:51:9 -new-name=Bar %s | FileCheck --check-prefix=CHECK4 %s

void macroInFunc() {
  #define VARNAME var
  int VARNAME;
}

// RUN: not clang-refactor-test rename-initiate -at=%s:58:19 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

void localVarArg() {
  int var; // CHECK5: rename local [[@LINE]]:7 -> [[@LINE]]:10
  BAR(var) = 0; // CHECK5-NEXT: rename local [[@LINE]]:7 -> [[@LINE]]:10
}

// RUN: clang-refactor-test rename-initiate -at=%s:65:7 -at=%s:66:7 -new-name=Bar %s | FileCheck --check-prefix=CHECK5 %s
