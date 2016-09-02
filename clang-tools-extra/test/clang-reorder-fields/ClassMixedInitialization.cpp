// RUN: clang-reorder-fields -record-name Foo -fields-order e,x,pi,s2,s1 %s -- -std=c++11 | FileCheck %s

class Foo {
public:
  Foo();

private:
  int x;              // CHECK:      {{^  double e = 2.71;}}
  const char *s1;     // CHECK-NEXT: {{^  int x;}}
  const char *s2;     // CHECK-NEXT: {{^  double pi = 3.14;}}
  double pi = 3.14;   // CHECK-NEXT: {{^  const char \*s2;}}
  double e = 2.71;    // CHECK-NEXT: {{^  const char \*s1;}}
};

Foo::Foo():
  x(12),      // CHECK:      {{^  x\(12\)}},
  s1("abc"),  // CHECK-NEXT: {{^  s2\("def"\)}},
  s2("def")   // CHECK-NEXT: {{^  s1\("abc"\)}}
{}

int main() {
  Foo foo;
  return 0;
}
