class A {
public:
  virtual void foo();
  virtual void bar();
};

class B : public A {
public:
  void foo() override;
};
