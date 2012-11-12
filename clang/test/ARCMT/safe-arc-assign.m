// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

void test12(id collection) {
  for (id x in collection) {
    x = 0;
    x = 0;
  }

  for (__strong id x in collection) {
    x = 0;
  }
}
