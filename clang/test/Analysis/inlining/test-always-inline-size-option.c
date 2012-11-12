// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-inline-max-stack-depth=3 -analyzer-config ipa-always-inline-size=3 -verify %s

void clang_analyzer_eval(int);
int nested5() {
  if (5 < 3)
    return 0;
  else
    if (3 == 3)
      return 0;
  return 0;
}
int nested4() {
  return nested5();
}
int nested3() {
  return nested4();
}
int nested2() {
  return nested3();
}
int nested1() {
  return nested2();
}

void testNested() {
  clang_analyzer_eval(nested1() == 0); // expected-warning{{TRUE}}
}

// Make sure we terminate a recursive path.
int recursive() {
  return recursive();
}
int callRecursive() {
  return recursive();
}

int mutuallyRecursive1();

int mutuallyRecursive2() {
  return mutuallyRecursive1();
}

int mutuallyRecursive1() {
  return mutuallyRecursive2();
}
int callMutuallyRecursive() {
  return mutuallyRecursive1();
}
