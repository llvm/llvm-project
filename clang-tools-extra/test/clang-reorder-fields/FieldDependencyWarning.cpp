// RUN: clang-reorder-fields -record-name bar::Foo -fields-order y,z,c,x %s -- 2>&1 | FileCheck --check-prefix=CHECK-MESSAGES %s
// FIXME: clang-reorder-fields should provide -verify mode to make writing these checks
// easier and more accurate, for now we follow clang-tidy's approach.

namespace bar {

struct Dummy {
  Dummy(int x, char c) : x(x), c(c) {}
  int x;
  char c;
};

class Foo {
public:
  Foo(int x, double y, char cin);
  Foo(int nx);
  Foo();
  int x;
  double y;
  char c;
  Dummy z;
};

static char bar(char c) {
  return c + 1;
}

Foo::Foo() : x(), y(), c(), z(0, 'a') {}

Foo::Foo(int x, double y, char cin) :  
  x(x),                 
  y(y),                 
  c(cin),               
  z(this->x, bar(c))    
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: reordering field x after z makes x uninitialized when used in init expression
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: reordering field c after z makes c uninitialized when used in init expression
{}

Foo::Foo(int nx) :
  x(nx),              
  y(x),
  c(0),            
  z(bar(bar(x)), c)     
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: reordering field x after y makes x uninitialized when used in init expression
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: reordering field x after z makes x uninitialized when used in init expression
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: reordering field c after z makes c uninitialized when used in init expression
{}

} // namespace bar

int main() {
  bar::Foo F(5, 12.8, 'c');
  return 0;
}
