// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,alpha.security.taint \
// RUN:   -mllvm -debug-only=taint-checker \
// RUN:   2>&1 | FileCheck %s

// REQUIRES: asserts

struct _IO_FILE;
typedef struct _IO_FILE FILE;
FILE *fopen(const char *fname, const char *mode);

void nested_call(void) {}

char *fgets(char *s, int n, FILE *fp) {
  nested_call();   // no-crash: we should not try adding taint to a non-existent argument.
  return (char *)0;
}

void top(const char *fname, char *buf) {
  FILE *fp = fopen(fname, "r");
  // CHECK:      PreCall<fopen(fname, "r")> prepares tainting arg index: -1
  // CHECK-NEXT: PostCall<fopen(fname, "r")> actually wants to taint arg index: -1

  if (!fp)
    return;

  (void)fgets(buf, 42, fp); // Trigger taint propagation.
  // CHECK-NEXT: PreCall<fgets(buf, 42, fp)> prepares tainting arg index: -1
  // CHECK-NEXT: PreCall<fgets(buf, 42, fp)> prepares tainting arg index: 0
  // CHECK-NEXT: PreCall<fgets(buf, 42, fp)> prepares tainting arg index: 2
  //
  // CHECK-NEXT: PostCall<fgets(buf, 42, fp)> actually wants to taint arg index: -1
  // CHECK-NEXT: PostCall<fgets(buf, 42, fp)> actually wants to taint arg index: 0
  // CHECK-NEXT: PostCall<fgets(buf, 42, fp)> actually wants to taint arg index: 2
}
