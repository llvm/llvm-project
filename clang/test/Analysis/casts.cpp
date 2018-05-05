// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -verify %s

bool PR14634(int x) {
  double y = (double)x;
  return !y;
}

bool PR14634_implicit(int x) {
  double y = (double)x;
  return y;
}

void intAsBoolAsSwitchCondition(int c) {
  switch ((bool)c) { // expected-warning {{switch condition has boolean value}}
  case 0:
    break;
  }

  switch ((int)(bool)c) { // no-warning
    case 0:
      break;
  }
}

int *&castToIntPtrLValueRef(char *p) {
  return (int *&)*(int *)p;
}
bool testCastToIntPtrLValueRef(char *p, int *s) {
  return castToIntPtrLValueRef(p) != s; // no-crash
}

int *&&castToIntPtrRValueRef(char *p) {
  return (int *&&)*(int *)p;
}
bool testCastToIntPtrRValueRef(char *p, int *s) {
  return castToIntPtrRValueRef(p) != s; // no-crash
}
