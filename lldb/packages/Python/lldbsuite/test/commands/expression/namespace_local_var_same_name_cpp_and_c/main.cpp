namespace error {
int x;
}

struct A {
  void foo() {
    int error = 1;

    return; // break here
  }
};

int main() {
  int error = 1;

  A a;

  a.foo();

  return 0; // break here
}
