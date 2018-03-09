class Base {
public:
  virtual ~Base() {}
  virtual void foo() {}
};

class Derived : public Base {
public:
  virtual void foo() {}
};

int main() {
  Derived d;
  Base *b = &d;
  return 0; // Set breakpoint here
}
