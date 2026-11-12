// RUN: %clangxx -fsanitize=returns-nonnull-attribute -w %s -O3 -o %t
// RUN: %run %t foo 2>&1 | FileCheck %s --check-prefix=NOERROR --allow-empty --implicit-check-not='runtime error'
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=returns-nonnull-attribute -fno-sanitize-recover=returns-nonnull-attribute -w %s -O3 -o %t.abort
// RUN: not %run %t.abort &> /dev/null

__attribute__((returns_nonnull)) char *foo(char *a);

char *foo(char *a) {
  // CHECK: nonnull.cpp:[[@LINE+2]]:3: runtime error: null pointer returned from function declared to never return null
  // CHECK-NEXT: nonnull.cpp:[[@LINE-4]]:16: note: returns_nonnull attribute specified here
  return a;
}

__attribute__((returns_nonnull)) char *bar(int x, char *a) {
  if (x > 10) {
    // CHECK: nonnull.cpp:[[@LINE+2]]:5: runtime error: null pointer returned from function declared to never return null
    // CHECK-NEXT: nonnull.cpp:[[@LINE-3]]:16: note: returns_nonnull attribute specified here
    return a;
  } else {
    // CHECK: nonnull.cpp:[[@LINE+2]]:5: runtime error: null pointer returned from function declared to never return null
    // CHECK-NEXT: nonnull.cpp:[[@LINE-7]]:16: note: returns_nonnull attribute specified here
    return a;
  }
}

int main(int argc, char **argv) {
  char *a = argv[1];

  foo(a);

  bar(20, a);

  // We expect to see a runtime error the first time we cover the "else"...
  bar(5, a);

  // ... but not a second time.
  // CHECK-NOT: runtime error
  bar(5, a);

  return 0;
}

// NOERROR-NOT: runtime error
