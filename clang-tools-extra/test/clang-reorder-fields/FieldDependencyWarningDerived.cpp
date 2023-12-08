// RUN: clang-reorder-fields -record-name bar::Derived -fields-order z,y %s -- 2>&1 | FileCheck --check-prefix=CHECK-MESSAGES %s
// FIXME: clang-reorder-fields should provide -verify mode to make writing these checks
// easier and more accurate, for now we follow clang-tidy's approach.

namespace bar {
struct Base {
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
  Base(),
  y(ny), 
  z(static_cast<char>(y)) 
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: reordering field y after z makes y uninitialized when used in init expression
{}

Derived::Derived(char nz) : 
  Base(),
  y(nz),
  // Check that base class fields are correctly ignored in reordering checks
  // x has field index 1 and so would improperly warn if this wasn't the case since the command for this file swaps field indexes 1 and 2
  z(x) 
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: reordering field x after z makes x uninitialized when used in init expression
{}

} // namespace bar
