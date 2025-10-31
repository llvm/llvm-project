// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false

struct S1 {
  char c;
};

struct S {
  int a;
  S1 b;
};

S GlobalS;

void doStuffP(const S *);
void doStuffR(const S &);

void uninit_val_p() {
  S s;
  doStuffP(&s); // expected-warning{{1st function call argument points to an uninitialized value (e.g., field: 'a')}}
}

void uninit_val_r() {
  S s;
  s.a = 0;
  doStuffR(s); // expected-warning{{1st function call argument references an uninitialized value (e.g., via the field chain: 'b.c')}}
}

S *uninit_new() {
  S *s = new S;
  doStuffP(s); // expected-warning{{1st function call argument points to an uninitialized value (e.g., field: 'a')}}
  return s;
}

void uninit_ctr() {
  S s = S();
  doStuffP(&s);
}

void uninit_init() {
  S s{};
  doStuffP(&s);
}

void uninit_init_val() {
  S s{1, {2}};
  doStuffP(&s);
}

void uninit_parm_ptr(S *s) {
  doStuffP(s);
}

void uninit_parm_val(S s) {
  doStuffP(&s);
}

void uninit_parm_ref(S &s) {
  doStuffP(&s);
}

void init_val() {
  S s;
  s.a = 1;
  s.b.c = 1;
  doStuffP(&s);
}

void uninit_global() {
  doStuffP(&GlobalS);
}

void uninit_static() {
  static S s;
  doStuffP(&s);
}
