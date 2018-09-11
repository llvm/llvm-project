// RUN: %clang_analyze_cc1 %s -analyzer-checker=core.NullDereference -analyzer-output=plist -o %t > /dev/null 2>&1
// RUN: cat %t | diff -u -w -I "<string>/" -I "<string>.:" -I "version" - %S/Inputs/expected-plists/inline-unique-reports.c.plist

static inline bug(int *p) {
  *p = 0xDEADBEEF;
}

void test_bug_1() {
  int *p = 0;
  bug(p);
}

void test_bug_2() {
  int *p = 0;
  bug(p);
}


