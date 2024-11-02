// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection \
// RUN:    -verify -analyzer-config eagerly-assume=false -std=c99 %s \
// RUN:    -Wno-implicit-function-declaration

int printf(const char *restrict,...);

void clang_analyzer_eval(int);
void clang_analyzer_dump(int*);

// Testing core functionality of the region store.
// radar://10127782
int compoundLiteralTest(void) {
    int index = 0;
    for (index = 0; index < 2; index++) {
        int thing = (int []){0, 1}[index];
        printf("thing: %i\n", thing);
    }
    return 0;
}

int compoundLiteralTest2(void) {
    int index = 0;
    for (index = 0; index < 3; index++) {
        int thing = (int [][3]){{0,0,0}, {1,1,1}, {2,2,2}}[index][index];
        printf("thing: %i\n", thing);
    }
    return 0;
}

int concreteOffsetBindingIsInvalidatedBySymbolicOffsetAssignment(int length,
                                                                 int i) {
  int values[length];
  values[i] = 4;
  return values[0]; // no-warning
}

struct X{
  int mem;
};
int initStruct(struct X *st);
int structOffsetBindingIsInvalidated(int length, int i){
  struct X l;
  initStruct(&l);
  return l.mem; // no-warning
}

void testConstraintOnRegionOffset(int *values, int length, int i){
  if (values[1] == 4) {
    values[i] = 5;
    clang_analyzer_eval(values[1] == 4);// expected-warning {{UNKNOWN}}
  }
}

int initArray(int *values);
void testConstraintOnRegionOffsetStack(int *values, int length, int i) {
  if (values[0] == 4) {
    initArray(values);
    clang_analyzer_eval(values[0] == 4);// expected-warning {{UNKNOWN}}
  }
}

int buffer[10];
void b(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C2x, conflicting with a subsequent definition}}
void missingPrototypeCallsiteMatchingArgsAndParams() {
  // expected-warning@+1 {{passing arguments to 'b' without a prototype is deprecated in all versions of C and is not supported in C2x}}
  b(&buffer);
}
void b(int *c) { // expected-note {{conflicting prototype is here}}
  clang_analyzer_dump(c); // expected-warning {{&Element{buffer,0 S64b,int}}}
  *c = 42; // no-crash
}

void c(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C2x, conflicting with a subsequent definition}}
void missingPrototypeCallsiteMismatchingArgsAndParams() {
  // expected-warning@+1 {{passing arguments to 'c' without a prototype is deprecated in all versions of C and is not supported in C2x}}
  c(&buffer, &buffer);
}
void c(int *c) { // expected-note {{conflicting prototype is here}}
  clang_analyzer_dump(c); // expected-warning {{Unknown}}
  *c = 42; // no-crash
}
