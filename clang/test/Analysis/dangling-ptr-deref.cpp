// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.ReportDanglingPtrDeref \
// RUN:   -analyzer-config cfg-lifetime=true -analyzer-output=text -verify %s

void test_case_one() {
  int *ptr = nullptr;
  // expected-note@+1 {{'num' is destroyed here}}
  {
    int num = 5;
    ptr = &num;
  }
  *ptr = 6;
  // expected-warning@-1 {{Use of 'num' after its lifetime ended}}
  // expected-note@-2    {{Use of 'num' after its lifetime ended}}
}

void test_case_two() {
  int *ptr_one = nullptr;
  int *ptr_two = nullptr;
  // expected-note@+2 {{'n' is destroyed here}}
  // expected-note@+1 {{'m' is destroyed here}}
  {
    int n = 1;
    int m = 2;
    ptr_one = &n;
    ptr_two = &m;
  }
  *ptr_one = 6;
  *ptr_two = 7; 
  // expected-warning@-2 {{Use of 'n' after its lifetime ended}}
  // expected-note@-3    {{Use of 'n' after its lifetime ended}}
  // expected-warning@-3 {{Use of 'm' after its lifetime ended}}
  // expected-note@-4    {{Use of 'm' after its lifetime ended}}
}

void escape(int *ptr);

void test_case_three() {
  int num = 5;
  int *ptr = &num;
  {
    *ptr = 6; // no-warning
  }
}

void test_case_four() {
  int *ptr = nullptr;
  // expected-note@+1 {{'num' is destroyed here}}
  {
    int num = 5;
    ptr = &num;
  }
  int i = *ptr;
  // expected-warning@-1 {{Use of 'num' after its lifetime ended}}
  // expected-note@-2    {{Use of 'num' after its lifetime ended}}
  i += i;
}

void test_case_five() {
  int *ptr = nullptr;
  for(int i = 0; i < 10; ++i) {
    ptr = &i;
  }
  escape(ptr); // no-warning
}

void test_case_six() {
  for(int i = 0; i < 10; ++i) {
    int *ptr = &i;
    escape(ptr); // no-warning
  }
}

void test_case_seven() {
  int *ptr = nullptr;
  // expected-note@+4 {{'i' is destroyed here}}
  // expected-note@+3 {{Loop condition is true.  Entering loop body}}
  // expected-note@+2 {{Assuming 'i' is >= 10}}
  // expected-note@+1 {{Loop condition is false. Execution continues on line 79}}
  for (int i = 0; i < 10; ++i) {
    ptr = &i;
    escape(ptr);
  }
  *ptr = 6;
  // expected-warning@-1 {{Use of 'i' after its lifetime ended}}
  // expected-note@-2    {{Use of 'i' after its lifetime ended}}
}
