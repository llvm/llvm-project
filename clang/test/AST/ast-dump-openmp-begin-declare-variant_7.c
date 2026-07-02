// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=C   %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=CXX %s -x c++


int OK_1(void);

#pragma omp begin declare variant match(implementation={vendor(intel)})
int OK_1(void) {
  return 1;
}
int OK_2(void) {
  return 1;
}
int not_OK(void) {
  return 1;
}
int OK_3(void) {
  return 1;
}
#pragma omp end declare variant

int OK_3(void);

int test(void) {
  // Should cause an error due to not_OK()
  return OK_1() + not_OK() + OK_3(); // CXX-error {{use of undeclared identifier 'not_OK'}} C-error {{call to undeclared function 'not_OK'; ISO C99 and later do not support implicit function declarations}}
}
