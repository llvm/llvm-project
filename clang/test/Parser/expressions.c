// RUN: %clang_cc1 -fsyntax-only -verify %s

void test1() {
  if (sizeof (int){ 1}) {}   // sizeof compound literal
  if (sizeof (int)) {}       // sizeof type

  (void)(int)4;   // cast.
  (void)(int){4}; // compound literal.

  int A = (struct{ int a;}){ 1}.a;
}

int test2(int a, int b) {
  return a ? (void)a,b : a;
}

int test3(int a, int b, int c) {
  return a = b = c;
}

int test4() {
  test4();
  return 0;
}

struct X0 { struct { struct { int c[10][9]; } b; } a; };

int test_offsetof() {
  (void)__builtin_offsetof(struct X0, a.b.c[4][5]);
  return 0;
}

void test_sizeof(){
        int arr[10];
        (void)sizeof arr[0];
        (void)sizeof(arr[0]);
        (void)sizeof(arr)[0];
}

// PR3418
int test_leading_extension() {
  __extension__ (*(char*)0) = 1; // expected-warning {{indirection of non-volatile null pointer}} \
                                 // expected-note {{consider using __builtin_trap}}
  return 0;
}

// PR3972
int test5(int);
int test6(void) { 
  return test5(      // expected-note {{to match}}
               test5(1)
                 ; // expected-error {{expected ')'}}
}

// PR8394
void test7() {
    ({} // expected-note {{to match}}
    ;   // expected-error {{expected ')'}}
}
