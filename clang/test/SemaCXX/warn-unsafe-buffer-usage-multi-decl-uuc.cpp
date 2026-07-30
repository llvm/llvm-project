// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s
void bar(int * param) {}

void foo1a() {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p = r;
  int tmp = p[9];  // expected-note{{used in buffer access here}}
  int *q;
  q = r;           // FIXME: we do not fix `q = r` here as the `.data()` fix-it is not generally correct
}

void uuc_if_body() {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} // expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' to 'std::span' to propagate bounds information between them}}
  if (true)
    p = r;
  p[5] = 4;  // expected-note{{used in buffer access here}}
}

void uuc_if_body1(bool flag) {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} // expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' to 'std::span' to propagate bounds information between them}}
  if (flag) {
    p = r;
  }
  p[5] = 4;  // expected-note{{used in buffer access here}}
}

void uuc_if_body2(bool flag) {
  int *r = new int[7];
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}} // expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' to 'std::span' to propagate bounds information between them}}
  if (flag) {
  } else {
    p = r;
  }

  p[5] = 4;  // expected-note{{used in buffer access here}}
}

void uuc_if_body2_ptr_init(bool flag) {
  int *r = new int[7];
  if (flag) {
  } else {
    int* p = r;  // expected-warning{{'p' is an unsafe pointer used for buffer access}} // expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'r' to 'std::span' to propagate bounds information between them}}
    p[5] = 4;  // expected-note{{used in buffer access here}}
  }
}

void uuc_if_cond_no_unsafe_op() {
  int *r = new int[7];
  int *p = new int[4];
  if ((p = r)) {
    int x = 0;
  }
}

void uuc_if_cond_no_unsafe_op1() {
  int *r = new int[7];
  int *p = new int[4];
  if (true) {
    int x = 0;
  } else if ((p = r))
    int y = 10;
}

void uuc_if_cond_unsafe_op() {
  int *r = new int[7];
  int *p = new int[4];  //expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if ((p = r)) {
    p[3] = 2;  // expected-note{{used in buffer access here}}
  }
}

void uuc_if_cond_unsafe_op1() {
  int *r = new int[7];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *p = new int[4];
  if ((p = r)) {
    r[3] = 2;  // expected-note{{used in buffer access here}}
  }
}

void uuc_if_cond_unsafe_op2() {
  int *r = new int[7];  // expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int *p = new int[4];  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if ((p = r)) {
    r[3] = 2;  // expected-note{{used in buffer access here}}
  }
  p[4] = 6;  // expected-note{{used in buffer access here}}
}

void uuc_call1() {
  int *w = new int[4];  // expected-warning{{'w' is an unsafe pointer used for buffer access}}
  int *y = new int[4];
  bar(w = y);
  w[5] = 0;  // expected-note{{used in buffer access here}}
}
