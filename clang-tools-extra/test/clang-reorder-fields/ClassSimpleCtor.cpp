// RUN: clang-reorder-fields -record-name Foo -fields-order s1,x,z,s2 %s -- | FileCheck %s

class Foo {
public:
  Foo();

private:
  int x;              // CHECK:      {{^  const char \*s1;}}
  const char *s1;     // CHECK-NEXT: {{^  int x;}}
  const char *s2;     // CHECK-NEXT: {{^  double z;}}
  double z;           // CHECK-NEXT: {{^  const char \*s2;}}
};

Foo::Foo():
  x(12),      // CHECK:      {{^  s1\("abc"\),}}
  s1("abc"),  // CHECK-NEXT: {{^  x\(12\),}}
  s2("def"),  // CHECK-NEXT: {{^  z\(3.14\),}}
  z(3.14)     // CHECK-NEXT: {{^  s2\("def"\)}}
{}

int main() {
  Foo foo;
  return 0;
}
