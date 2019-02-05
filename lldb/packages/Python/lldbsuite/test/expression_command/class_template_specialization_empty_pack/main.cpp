template <typename N, class... P>
struct A {
    int foo() { return 1;}
};

int main() {
  A<int> b;
  return b.foo(); // break here
}
