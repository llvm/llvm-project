// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_1 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_2 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_3 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_4 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_5 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wconditional-scope -DTEST_6 %s

int *get_something();
int *get_something_else();
int *get_something_else_again();
int *get_something_else_again_now();

#ifdef TEST_1

int test() {
  if (int *ptr = get_something()) {
    return ptr[0] * ptr[0];
  }
  // expected-warning@+2{{variable 'ptr' declared in 'if' block is always false or null here}}
  else if (int *ptr2 = get_something_else()) {
    return ptr[0] * ptr2[0];
  }
  // expected-warning@+3{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr2' declared in 'if' block is always false or null here}}
  else if (int* ptr3 = get_something_else_again()) {
    return ptr[0] * ptr2[0] * ptr3[0];	
  }
  // expected-warning@+4{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+3{{variable 'ptr2' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr3' declared in 'if' block is always false or null here}}
  else if (int *ptr4 = get_something_else_again_now()) {
    return ptr[0] * ptr2[0] * ptr3[0] * ptr4[0];
  }
  else {
    return -1;
  }
}

#endif

#ifdef TEST_2

int test() {
  if (int *ptr = get_something()) {
    return ptr[0] * ptr[0];
  }
  else if (int *ptr2 = get_something_else()) {
    return ptr2[0] * ptr2[0];
  }
  // expected-warning@+3{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr2' declared in 'if' block is always false or null here}}
  else if (int* ptr3 = get_something_else_again()) {
    return ptr[0] * ptr2[0] * ptr3[0];
  }
  else {
    return -1;
  }
}

#endif

#ifdef TEST_3

int test() {
  if (int *ptr = get_something()) {
    return ptr[0] * ptr[0];
  }
  else if (int *ptr2 = get_something_else()) {
    return ptr2[0] * ptr2[0];
  }
  // expected-warning@+3{{variable 'ptr2' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr2' declared in 'if' block is always false or null here}}
  else if (int* ptr3 = get_something_else_again()) {
    return ptr2[0] * ptr2[0] * ptr3[0];
  }
  else {
    return -1;
  }
}

#endif

#ifdef TEST_4

int test() {
  int x = 10; 

  if (x == 30) {
    return x;
  }
  else if (int *ptr = get_something()) {
    return ptr[0];
  }
  // expected-warning@+3{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr' declared in 'if' block is always false or null here}}
  else if (x == 20) {
    return ptr[0] * ptr[0];
  }
  // expected-warning@+2{{variable 'ptr' declared in 'if' block is always false or null here}}
  else if (int* ptr2 = get_something_else()) {
    return ptr[0] * ptr2[0];  
  }
  // expected-warning@+3{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr2' declared in 'if' block is always false or null here}}
  else if (int *ptr3 = get_something_else_again()) {
    return ptr[0] * ptr2[0] * ptr3[0];
  }
  else {
    return -1; 
  }
}

#endif

#ifdef TEST_5

void test() {
  // expected-no-diagnostics
  if (bool x = get_something()) {}
  else {
    {   
      bool x = get_something_else();
      if (x) {}
    }   
  }
}

#endif

#ifdef TEST_6

int test() {
  if (int *ptr = get_something()) {
    return ptr[0];
  }
  else if (int *ptr = get_something_else()) {
    return ptr[0] * ptr[0];
  }
  // expected-warning@+3{{variable 'ptr' declared in 'if' block is always false or null here}}
  // expected-warning@+2{{variable 'ptr' declared in 'if' block is always false or null here}}
  else {
    return ptr[0] * ptr[0];
  }
}

#endif
