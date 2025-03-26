// RUN: %check_clang_tidy %s readability-ambiguous-smartptr-reset-call %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-ambiguous-smartptr-reset-call.SmartPointers: "::std::unique_ptr;::other_ptr"}}' \
// RUN: --fix-notes -- -I %S/../modernize/Inputs/smart-ptr

#include "unique_ptr.h"
#include "shared_ptr.h"

template <typename T>
struct other_ptr {
  T& operator*() const;
  T* operator->() const;
  void reset();
};

struct Resettable {
  void reset();
  void doSomething();
};

void Positive() {
  std::unique_ptr<Resettable> u;
  u.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u = nullptr;
  u->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u).reset();

  other_ptr<Resettable> s;
  s.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s = nullptr;
  s->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s).reset();
}

void Negative() {
  std::shared_ptr<Resettable> s_ptr;
  s_ptr.reset();
  s_ptr->reset();
  s_ptr->doSomething();

  std::unique_ptr<Resettable> u_ptr;
  u_ptr.reset(nullptr);
  u_ptr->doSomething();
}
