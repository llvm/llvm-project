// RUN: %clang_cc1 -fsyntax-only -Wno-dangling-gsl -Wreturn-stack-address -verify %s

struct [[gsl::Owner(int)]] MyIntOwner {
  MyIntOwner();
  int &operator*();
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  MyIntPointer(int *p = nullptr);
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
};

int &f() {
  int i;
  return i; // expected-warning {{reference to stack memory associated with local variable 'i' returned}}
}

MyIntPointer g() {
  MyIntOwner o;
  return o; // No warning, it is disabled.
}

void h(MyIntPointer p) {
  p = MyIntOwner(); //  No warning, it is disabled.
}
