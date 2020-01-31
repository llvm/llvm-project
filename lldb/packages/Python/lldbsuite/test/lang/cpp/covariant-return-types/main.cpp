struct OtherBase {
  // Allow checking actual type from the test by giving
  // this class and the subclass unique values here.
  virtual const char *value() { return "base"; }
};
struct OtherDerived : public OtherBase {
  const char *value() override { return "derived"; }
};

// Those have to be globals as they would be completed if they
// are members (which would make this test always pass).
OtherBase other_base;
OtherDerived other_derived;

struct Base {
  // Function with covariant return type that is same class.
  virtual Base* getPtr() { return this; }
  virtual Base& getRef() { return *this; }
  // Function with covariant return type that is a different class.
  virtual OtherBase* getOtherPtr() { return &other_base; }
  virtual OtherBase& getOtherRef() { return other_base; }
};

struct Derived : public Base {
  Derived* getPtr() override { return this; }
  Derived& getRef() override { return *this; }
  OtherDerived* getOtherPtr() override { return &other_derived; }
  OtherDerived& getOtherRef() override { return other_derived; }
};

int main() {
  Derived derived;
  Base base;
  Base *base_ptr_to_derived = &derived;
  (void)base_ptr_to_derived->getPtr();
  (void)base_ptr_to_derived->getRef();
  (void)base_ptr_to_derived->getOtherPtr();
  (void)base_ptr_to_derived->getOtherRef();
  return 0; // break here
}
