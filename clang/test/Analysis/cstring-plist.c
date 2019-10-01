// RUN: rm -f %t
// RUN: %clang_analyze_cc1 -fblocks \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=unix.cstring.NullArg \
// RUN:   -analyzer-disable-checker=alpha.unix.cstring.OutOfBounds \
// RUN:   -analyzer-output=plist -o %t %s
// RUN: FileCheck -input-file %t %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
char *strncpy(char *restrict s1, const char *restrict s2, size_t n);



void cstringchecker_bounds_nocrash() {
  char *p = malloc(2);
  strncpy(p, "AAA", sizeof("AAA")); // we don't expect warning as the checker is disabled
  free(p);
}

// CHECK: <key>diagnostics</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>
// CHECK-NEXT: <key>files</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>
// CHECK-NEXT: </dict>
// CHECK-NEXT: </plist>
