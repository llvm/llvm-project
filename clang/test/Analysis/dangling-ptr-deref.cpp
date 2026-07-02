// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.ReportDanglingPtrDeref \
// RUN:   -analyzer-config cfg-lifetime=true -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.ReportDanglingPtrDeref \
// RUN:   -analyzer-config c++-container-inlining=false -analyzer-config cfg-lifetime=true -verify %s

void test_case_one() {
  int *ptr = nullptr;
  {
    int num = 5;
    ptr = &num;
  }
  *ptr = 6; // expected-warning {{Use of 'num' after its lifetime ended}}
}

void test_case_two() {
  int *ptr_one = nullptr;
  int *ptr_two = nullptr;
  {
    int n = 1;
    int m = 2;
    ptr_one = &n;
    ptr_two = &m;
  }
  *ptr_one = 6; // expected-warning {{Use of 'n' after its lifetime ended}}
  *ptr_two = 7; // expected-warning {{Use of 'm' after its lifetime ended}}
}

void test_case_three() {
  int num = 5;
  int *ptr = &num;
  {
    *ptr = 6; // no-warning
  }
}

void test_case_four() {
  int *ptr = nullptr;
  {
    int num = 5;
    ptr = &num;
  }
  int i = *ptr; // expected-warning {{Use of 'num' after its lifetime ended}}
  i += i;
}

