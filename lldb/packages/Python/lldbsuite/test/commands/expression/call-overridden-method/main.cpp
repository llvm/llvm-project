class Base {
public:
  virtual ~Base() {}
  virtual int foo() { return 1; }
};

class Derived : public Base {
public:
  virtual int foo() { return 2; }
};

int main() {
  Base realbase;
  realbase.foo();
  Derived d;
  Base *b = &d;
  return 0; // Set breakpoint here
}
