// RUN: %clang_analyze_cc1 -analyzer-checker core,unix.Malloc -analyzer-output=plist -o %t.plist -verify %s
// RUN: FileCheck --input-file=%t.plist %s

void free(void *);
void (*fnptr)(int);
void foo(void) {
  free((void *)fnptr);
  // expected-warning@-1{{Argument to 'free()' is a function pointer}}
  // expected-warning@-2{{attempt to call free on non-heap object '(void *)fnptr'}}
}

// Make sure the bug category is correct.
// CHECK: <key>category</key><string>Memory error</string>
