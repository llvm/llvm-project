class Foo {
public:
  Foo(int i, double d) : m_i(i), m_d(d){};

private:
  int m_i;
  int m_d;
};

int main(int argc, char **argv) {
  Foo foo(1, 2.22);
  return 0;
}
