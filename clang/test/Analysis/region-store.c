// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection -verify %s

int printf(const char *restrict,...);

// Testing core functionality of the region store.
// radar://10127782
int compoundLiteralTest() {
    int index = 0;
    for (index = 0; index < 2; index++) {
        int thing = (int []){0, 1}[index];
        printf("thing: %i\n", thing);
    }
    return 0;
}

int compoundLiteralTest2() {
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

void clang_analyzer_eval(int);
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
