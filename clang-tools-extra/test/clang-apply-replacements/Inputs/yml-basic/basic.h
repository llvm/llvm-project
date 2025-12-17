#ifndef BASIC_H
#define BASIC_H


class Parent {
public:
  virtual void func() {}
};

class Derived : public Parent {
public:
  virtual void func() {}
  // CHECK: virtual void func() override {}
};

extern void ext(int (&)[5], const Parent &);

void func(int t) {
  int ints[5];
  for (unsigned i = 0; i < 5; ++i) {
    int &e = ints[i];
    e = t;
    // CHECK: for (auto & elem : ints) {
    // CHECK-NEXT: elem = t;
  }

  Derived d;

  ext(ints, d);
}

#endif // BASIC_H
