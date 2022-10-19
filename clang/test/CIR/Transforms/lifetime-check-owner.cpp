// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="history=all;remarks=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

struct [[gsl::Owner(int)]] MyIntOwner {
  int val;
  MyIntOwner(int v) : val(v) {}
  void changeInt(int i);
  int &operator*();
  int read() const;
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  int *ptr;
  MyIntPointer(int *p = nullptr) : ptr(p) {}
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
  int read() { return *ptr; }
};

void yolo() {
  MyIntPointer p;
  {
    MyIntOwner o(1);
    p = o;
    *p = 3; // expected-remark {{pset => { o' }}}
  }       // expected-note {{pointee 'o' invalidated at end of scope}}
  *p = 4; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
}

void yolo2() {
  MyIntPointer p;
  MyIntOwner o(1);
  p = o;
  (void)o.read();
  o.changeInt(42); // expected-note {{invalidated by non-const use of owner type}}
  (void)p.read(); // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
}
