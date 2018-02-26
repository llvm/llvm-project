// RUN: %clang_analyze_cc1 -analyzer-checker core,unix.Malloc -analyzer-output=plist -o %t.plist -verify %s
// RUN: FileCheck --input-file=%t.plist %s

void free(void *);
void (*fnptr)(int);
void foo() {
  free((void *)fnptr); // expected-warning{{Argument to free() is a function pointer}}
}

// Make sure the bug category is correct.
// CHECK: <key>category</key><string>Memory error</string>
