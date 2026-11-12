// RUN: %check_clang_tidy -std=c++20 %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr -D CXX_14_17=1

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

template <class T>
struct TemplatedWithMultipleDestructors {
  ~TemplatedWithMultipleDestructors() requires(sizeof(T) > 1) = default;    // selected for TemplatedWithMultipleDestructors<int>
private:
  ~TemplatedWithMultipleDestructors() requires(sizeof(T) == 1) = default;   // selected for TemplatedWithMultipleDestructors<char>
};


struct WithFriend {
  friend struct std::default_delete<WithFriend>;

private:
  ~WithFriend() = default;
};

void check_new() {
  std::unique_ptr<WithFriend> p(new WithFriend());
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<WithFriend> p = std::make_unique<WithFriend>();
}

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

  std::unique_ptr<TemplatedWithMultipleDestructors<int>> t0;
  t0.reset(new TemplatedWithMultipleDestructors<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead
  // CHECK-FIXES: t0 = std::make_unique<TemplatedWithMultipleDestructors<int>>();

  std::unique_ptr<TemplatedWithMultipleDestructors<char>> t1;
  t1.reset(new TemplatedWithMultipleDestructors<char>());
}
