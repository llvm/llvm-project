struct Foo {
  int a;
  int b;
};

struct Foo *GetAFoo() {
  return 0;
}

int main() {
  return GetAFoo()->b;
}
