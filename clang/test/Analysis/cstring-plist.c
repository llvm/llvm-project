// RUN: rm -f %t
// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,unix.Malloc,unix.cstring.NullArg -analyzer-disable-checker=alpha.unix.cstring.OutOfBounds -analyzer-output=plist -analyzer-config path-diagnostics-alternate=false -o %t %s
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
// CHECK-NEXT: </dict>
// CHECK-NEXT: </plist>
