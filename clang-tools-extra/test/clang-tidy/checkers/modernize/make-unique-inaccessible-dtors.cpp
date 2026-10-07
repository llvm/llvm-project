// RUN: %check_clang_tidy -std=c++14,c++17 %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr -D CXX_14_17=1

#include "unique_ptr.h"
// CHECK-FIXES: #include <memory>

struct Base {
};

struct ChildWithPublicDestructor: Base {
public:
  ~ChildWithPublicDestructor();
};

struct ChildWithProtectedDestructor: Base {
protected:
  ~ChildWithProtectedDestructor();
};

struct ChildWithPrivateDestructor: Base {
private:
  ~ChildWithPrivateDestructor();
};

void check_reset() {
  std::unique_ptr<Base> p;
  p.reset(new ChildWithPublicDestructor());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_unique instead
  // CHECK-FIXES: p = std::make_unique<ChildWithPublicDestructor>();
  p.reset(new ChildWithProtectedDestructor());
  p.reset(new ChildWithPrivateDestructor());
}
