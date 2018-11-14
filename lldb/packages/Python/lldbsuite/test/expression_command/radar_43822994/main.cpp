enum class Foo {
  FooBar = 42
};

int main(int argc, const char **argv) {
  Foo f = Foo::FooBar;
  bool b1 = f == Foo::FooBar;
  return 0; // Set break point at this line.
}
