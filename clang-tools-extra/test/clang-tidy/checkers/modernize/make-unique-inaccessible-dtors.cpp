// RUN: %check_clang_tidy -std=c++14,c++17 %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr -D CXX_14_17=1

#include "unique_ptr.h"
// CHECK-FIXES: #include <memory>

struct Base {
};

struct ChildWithPublicDestructor: Base {
public:
  ~ChildWithPublicDestructor();
};

using ChildWithPublicDestructorAlias = ChildWithPublicDestructor;

template<class T>
struct TemplatedChildWithPublicDestructor: Base {
public:
  ~TemplatedChildWithPublicDestructor();
};

struct ChildWithProtectedDestructor: Base {
protected:
  ~ChildWithProtectedDestructor();
};

struct ChildWithPrivateDestructor: Base {
private:
  ~ChildWithPrivateDestructor();
};

struct ChildWithDeletedDestructor: Base {
  ~ChildWithDeletedDestructor() = delete;
};

using ChildWithDeletedDestructorAlias = ChildWithDeletedDestructor;

template<class T>
struct TemplatedChildWithDeletedDestructor: Base {
  ~TemplatedChildWithDeletedDestructor() = delete;
};

void check_reset() {
  std::unique_ptr<Base> p;
  p.reset(new ChildWithPublicDestructor());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_unique instead
  // CHECK-FIXES: p = std::make_unique<ChildWithPublicDestructor>();
  p.reset(new ChildWithPublicDestructorAlias());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_unique instead
  // CHECK-FIXES: p = std::make_unique<ChildWithPublicDestructorAlias>();
  p.reset(new TemplatedChildWithPublicDestructor<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_unique instead
  // CHECK-FIXES: p = std::make_unique<TemplatedChildWithPublicDestructor<int>>();
  p.reset(new ChildWithProtectedDestructor());
  p.reset(new ChildWithPrivateDestructor());
  p.reset(new ChildWithDeletedDestructor());
  p.reset(new ChildWithDeletedDestructorAlias());
  p.reset(new TemplatedChildWithDeletedDestructor<int>());
}
