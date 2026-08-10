// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-inline-max-stack-depth=3 -analyzer-config ipa-always-inline-size=3 -verify %s

void clang_analyzer_eval(int);
int nested5(void) {
  if (5 < 3)
    return 0;
  else
    if (3 == 3)
      return 0;
  return 0;
}
int nested4(void) {
  return nested5();
}
int nested3(void) {
  return nested4();
}
int nested2(void) {
  return nested3();
}
int nested1(void) {
  return nested2();
}

void testNested(void) {
  clang_analyzer_eval(nested1() == 0); // expected-warning{{TRUE}}
}

// Make sure we terminate a recursive path.
int recursive(void) {
  return recursive();
}
int callRecursive(void) {
  return recursive();
}

int mutuallyRecursive1(void);

int mutuallyRecursive2(void) {
  return mutuallyRecursive1();
}

int mutuallyRecursive1(void) {
  return mutuallyRecursive2();
}
int callMutuallyRecursive(void) {
  return mutuallyRecursive1();
}
