// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring -analyzer-output=text %s 2>&1 | FileCheck %s

// This test verifies argument source range highlighting.
// Otherwise we've no idea which of the arguments is null.
// These days we actually also have it in the message,
// but the range is still great to have.

char *strcpy(char *, const char *);

void foo() {
  char *a = 0, *b = 0;
  strcpy(a, b);
}

// CHECK: warning: Null pointer passed as 1st argument to string copy function
// CHECK-NEXT: strcpy(a, b);
// CHECK-NEXT: ^      ~
