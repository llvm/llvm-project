// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s
struct __attribute__((warn_unused)) Test {
  Test();
  ~Test();
  void use();
};

struct TestNormal {
  TestNormal();
};

int main(void) {
  Test unused;         // expected-warning {{unused variable 'unused'}}
  Test used;
  TestNormal normal;
  used.use();

  int i __attribute__((warn_unused)) = 12; // expected-warning {{'warn_unused' attribute only applies to structs, unions, and classes}}
  return i;
}
