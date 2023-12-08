// RUN: %clang_analyze_cc1 -analyzer-checker=core.NullDereference -analyzer-output=text -fno-caret-diagnostics %s 2>&1 | FileCheck %s

void testA(void) {
  int *p = 0;
  *p = 1;

  // CHECK-LABEL: text-diagnostics.c:{{.*}}:6: warning: Dereference of null pointer (loaded from variable 'p')
  // CHECK-NEXT: text-diagnostics.c:[[@LINE-4]]:3: note: 'p' initialized to a null pointer value
  // CHECK-NEXT: text-diagnostics.c:[[@LINE-4]]:6: note: Dereference of null pointer (loaded from variable 'p')
}

void testB(int *q) {
  if (q)
    return;
  *q = 1;

  // CHECK-LABEL: text-diagnostics.c:{{.*}}:6: warning: Dereference of null pointer (loaded from variable 'q')
  // CHECK-NEXT: text-diagnostics.c:[[@LINE-5]]:7: note: Assuming 'q' is null
  // CHECK-NEXT: text-diagnostics.c:[[@LINE-6]]:3: note: Taking false branch
  // CHECK-NEXT: text-diagnostics.c:[[@LINE-5]]:6: note: Dereference of null pointer (loaded from variable 'q')
}
