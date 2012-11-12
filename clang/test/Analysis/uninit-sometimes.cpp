// RUN: %clang_cc1 -std=gnu++11 -Wsometimes-uninitialized -verify %s
// RUN: %clang_cc1 -std=gnu++11 -Wsometimes-uninitialized -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

bool maybe();

int test_if_false(bool b) {
  int x; // expected-note {{variable}}
  if (b) // expected-warning {{whenever 'if' condition is false}} \
         // expected-note {{remove the 'if' if its condition is always true}}
    x = 1;
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{8:3-10:5}:""
// CHECK: fix-it:"{{.*}}":{7:8-7:8}:" = 0"


int test_if_true(bool b) {
  int x; // expected-note {{variable}}
  if (b) {} // expected-warning {{whenever 'if' condition is true}} \
            // expected-note {{remove the 'if' if its condition is always false}}
  else x = 1;
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{20:3-22:8}:""
// CHECK: fix-it:"{{.*}}":{19:8-19:8}:" = 0"


int test_while_false(bool b) {
  int x; // expected-note {{variable}}
  while (b) { // expected-warning {{whenever 'while' loop exits because its condition is false}} \
              // expected-note {{remove the condition if it is always true}}
    if (maybe()) {
      x = 1;
      break;
    }
  };
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{32:10-32:11}:"true"
// CHECK: fix-it:"{{.*}}":{31:8-31:8}:" = 0"


int test_while_true(bool b) {
  int x; // expected-note {{variable}}
  while (b) { // expected-warning {{whenever 'while' loop is entered}} \
              // expected-note {{remove the condition if it is always false}}
label:
    return x; // expected-note {{uninitialized use}}
  }
  x = 0;
  goto label;
}

// CHECK: fix-it:"{{.*}}":{48:10-48:11}:"false"
// CHECK: fix-it:"{{.*}}":{47:8-47:8}:" = 0"


int test_do_while_false(bool b) {
  int x; // expected-note {{variable}}
  do {
    if (maybe()) {
      x = 1;
      break;
    }
  } while (b); // expected-warning {{whenever 'do' loop exits because its condition is false}} \
               // expected-note {{remove the condition if it is always true}}
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{68:12-68:13}:"true"
// CHECK: fix-it:"{{.*}}":{62:8-62:8}:" = 0"


int test_do_while_true(bool b) {
  int x; // expected-note {{variable}}
goto label2;
  do {
label1:
    return x; // expected-note {{uninitialized use}}
label2: ;
  } while (b); // expected-warning {{whenever 'do' loop condition is true}} \
               // expected-note {{remove the condition if it is always false}}
  x = 0;
  goto label1;
}

// CHECK: fix-it:"{{.*}}":{84:12-84:13}:"false"
// CHECK: fix-it:"{{.*}}":{78:8-78:8}:" = 0"


int test_for_false(int k) {
  int x; // expected-note {{variable}}
  for (int n = 0;
       n < k; // expected-warning {{whenever 'for' loop exits because its condition is false}} \
              // expected-note {{remove the condition if it is always true}}
       ++n) {
    if (maybe()) {
      x = n;
      break;
    }
  }
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{97:8-97:13}:""
// CHECK: fix-it:"{{.*}}":{95:8-95:8}:" = 0"


int test_for_true(int k) {
  int x; // expected-note {{variable}}
  int n = 0;
  for (;
       n < k; // expected-warning {{whenever 'for' loop is entered}} \
              // expected-note {{remove the condition if it is always false}}
       ++n) {
label:
    return x; // expected-note {{uninitialized use}}
  }
  x = 1;
  goto label;
}

// CHECK: fix-it:"{{.*}}":{116:8-116:13}:"false"
// CHECK: fix-it:"{{.*}}":{113:8-113:8}:" = 0"


int test_for_range_false(int k) {
  int arr[3] = { 1, 2, 3 };
  int x;
  for (int &a : arr) { // no-warning, condition was not explicitly specified
    if (a == k) {
      x = &a - arr;
      break;
    }
  }
  return x;
}





int test_for_range_true(int k) {
  int arr[3] = { 1, 2, 3 };
  int x;
  for (int &a : arr) { // no-warning
    goto label;
  }
  x = 0;
label:
  return x;
}





int test_conditional_false(int k) {
  int x; // expected-note {{variable}}
  (void)(
      maybe() // expected-warning {{whenever '?:' condition is false}} \
              // expected-note {{remove the '?:' if its condition is always true}}
      ? x = 1 : 0);
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{164:7-166:9}:""
// CHECK: fix-it:"{{.*}}":{166:14-166:18}:""
// CHECK: fix-it:"{{.*}}":{162:8-162:8}:" = 0"

int test_conditional_true(int k) {
  int x; // expected-note {{variable}}
  (void)(
      maybe() // expected-warning {{whenever '?:' condition is true}} \
              // expected-note {{remove the '?:' if its condition is always false}}
      ? 0 : x = 1);
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{177:7-179:13}:""
// CHECK: fix-it:"{{.*}}":{175:8-175:8}:" = 0"


int test_logical_and_false(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-warning {{whenever '&&' condition is false}} \
          // expected-note {{remove the '&&' if its condition is always true}}
      && (x = 1);
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{189:3-191:9}:""
// CHECK: fix-it:"{{.*}}":{188:8-188:8}:" = 0"


int test_logical_and_true(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-warning {{whenever '&&' condition is true}} \
          // expected-note {{remove the '&&' if its condition is always false}}
      && ({ goto skip_init; 0; });
  x = 1;
skip_init:
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{201:3-203:34}:"false"
// CHECK: fix-it:"{{.*}}":{200:8-200:8}:" = 0"


int test_logical_or_false(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-warning {{whenever '||' condition is false}} \
          // expected-note {{remove the '||' if its condition is always true}}
      || ({ goto skip_init; 0; });
  x = 1;
skip_init:
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{215:3-217:34}:"true"
// CHECK: fix-it:"{{.*}}":{214:8-214:8}:" = 0"


int test_logical_or_true(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-warning {{whenever '||' condition is true}} \
          // expected-note {{remove the '||' if its condition is always false}}
      || (x = 1);
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{229:3-231:9}:""
// CHECK: fix-it:"{{.*}}":{228:8-228:8}:" = 0"


int test_switch_case(int k) {
  int x; // expected-note {{variable}}
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1: // expected-warning {{whenever switch case is taken}}
    break;
  }
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{240:8-240:8}:" = 0"



int test_switch_default(int k) {
  int x; // expected-note {{variable}}
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1:
    x = 1;
    break;
  default: // expected-warning {{whenever switch default is taken}}
    break;
  }
  return x; // expected-note {{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{256:8-256:8}:" = 0"



int test_switch_suppress_1(int k) {
  int x;
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1:
    x = 1;
    break;
  }
  return x; // no-warning
}





int test_switch_suppress_2(int k) {
  int x;
  switch (k) {
  case 0:
  case 1:
    switch (k) {
    case 0:
      return 0;
    case 1:
      return 1;
    }
  case 2:
  case 3:
    x = 1;
  }
  return x; // no-warning
}





int test_multiple_notes(int k) {
  int x; // expected-note {{variable}}
  if (k > 0) {
    if (k == 5)
      x = 1;
    else if (k == 2) // expected-warning {{whenever 'if' condition is false}} \
                     // expected-note {{remove the 'if' if its condition is always true}}
      x = 2;
  } else {
    if (k == -5)
      x = 3;
    else if (k == -2) // expected-warning {{whenever 'if' condition is false}} \
                      // expected-note {{remove the 'if' if its condition is always true}}
      x = 4;
  }
  return x; // expected-note 2{{uninitialized use}}
}

// CHECK: fix-it:"{{.*}}":{324:10-326:7}:""
// CHECK: fix-it:"{{.*}}":{318:10-320:7}:""
// CHECK: fix-it:"{{.*}}":{314:8-314:8}:" = 0"

int test_no_false_positive_1(int k) {
  int x;
  if (k)
    x = 5;
  while (!k)
    maybe();
  return x;
}





int test_no_false_positive_2() {
  int x;
  bool b = false;
  if (maybe()) {
    x = 5;
    b = true;
  }
  return b ? x : 0;
}


// FIXME: In this case, the variable is used uninitialized whenever the
// function's entry block is reached. Produce a diagnostic saying that
// the variable is uninitialized the first time it is used.
void test_null_pred_succ() {
  int x;
  if (0)
    foo: x = 0;
  if (x)
    goto foo;
}




void foo();
int PR13360(bool b) {
  int x; // expected-note {{variable}}
  if (b) { // expected-warning {{variable 'x' is used uninitialized whenever 'if' condition is true}} expected-note {{remove}}
    do {
      foo();
    } while (0);
  } else {
    x = 1;
  }
  return x; // expected-note {{uninitialized use occurs here}}
}

// CHECK: fix-it:"{{.*}}":{376:3-380:10}:""
// CHECK: fix-it:"{{.*}}":{375:8-375:8}:" = 0"
