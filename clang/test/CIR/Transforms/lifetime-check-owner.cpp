// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

struct [[gsl::Owner(int)]] MyIntOwner {
  int val;
  MyIntOwner(int v) : val(v) {}
  int &operator*();
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  int *ptr;
  MyIntPointer(int *p = nullptr) : ptr(p) {}
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
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