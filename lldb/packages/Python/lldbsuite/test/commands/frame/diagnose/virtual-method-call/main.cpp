class Foo {
public:
  int a;
  int b;
  virtual int Sum() { return a + b; }
};

struct Foo *GetAFoo() {
  return (struct Foo*)0;
}

int main() {
  struct Foo *foo = GetAFoo();
  return foo->Sum();
}

