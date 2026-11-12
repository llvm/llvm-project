class Base {
public:
  virtual void foo() {}
};

class Derived : public Base {
public:
  void foo() override {}
};
