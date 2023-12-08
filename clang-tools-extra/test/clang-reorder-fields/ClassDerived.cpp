// RUN: clang-reorder-fields -record-name bar::Derived -fields-order z,y %s -- | FileCheck %s

namespace bar {
class Base {
public:
  Base(int nx, int np) : x(nx), p(np) {}
  int x;
  int p;
};


class Derived : public Base {
public:
  Derived(long ny);
  Derived(char nz);
private:
  long y;
  char z;
};

Derived::Derived(long ny) : 
    Base(ny, 0),
    y(ny),                   // CHECK:       {{^  z\(static_cast<char>\(ny\)\),}}
    z(static_cast<char>(ny)) // CHECK-NEXT:  {{^  y\(ny\)}}
{}

Derived::Derived(char nz) : 
    Base(1, 2),
    y(nz),  // CHECK:       {{^  z\(x\),}}
    z(x)    // CHECK-NEXT:  {{^  y\(nz\)}}
{}

} // namespace bar
