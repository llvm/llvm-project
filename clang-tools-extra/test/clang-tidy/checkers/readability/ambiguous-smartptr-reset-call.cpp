// RUN: %check_clang_tidy %s readability-ambiguous-smartptr-reset-call %t --fix-notes -- -I %S/../modernize/Inputs/smart-ptr

#include "unique_ptr.h"
#include "shared_ptr.h"

template <typename T>
struct non_default_reset_ptr {
  T& operator*() const;
  T* operator->() const;
  void reset(T* p);
};

struct Resettable {
  void reset();
  void doSomething();
};

struct ResettableWithParam {
  void reset(int a);
  void doSomething();
};

struct ResettableWithDefaultParams {
  void reset(int a = 0, double b = 0.0);
  void doSomething();
};

struct NonResettable {
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

  std::shared_ptr<Resettable> s;
  s.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s = nullptr;
  s->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s).reset();

  std::unique_ptr<std::unique_ptr<int>> uu_ptr;
  uu_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: uu_ptr = nullptr;
  uu_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*uu_ptr).reset();

  std::unique_ptr<std::shared_ptr<int>> su_ptr;
  su_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: su_ptr = nullptr;
  su_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*su_ptr).reset();

  std::unique_ptr<ResettableWithDefaultParams> rd;
  rd.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: rd = nullptr;
  rd->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*rd).reset();

  std::unique_ptr<std::shared_ptr<std::unique_ptr<Resettable>>> nested_ptr;
  nested_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: nested_ptr = nullptr;
  nested_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*nested_ptr).reset();
  (*nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: (*nested_ptr) = nullptr;
  (*nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*(*nested_ptr)).reset();
  (**nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: (**nested_ptr) = nullptr;
  (**nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*(**nested_ptr)).reset();
}

void Negative() {
  std::unique_ptr<Resettable> u_ptr;
  u_ptr.reset(nullptr);
  u_ptr->doSomething();

  std::shared_ptr<Resettable> s_ptr;
  s_ptr.reset(nullptr);
  s_ptr->doSomething();
  
  Resettable* raw_ptr;
  raw_ptr->reset();
  raw_ptr->doSomething();

  Resettable resettable;
  resettable.reset();
  resettable.doSomething();
  
  std::unique_ptr<ResettableWithParam> u_ptr_param;
  u_ptr_param.reset();
  u_ptr_param.reset(nullptr);
  u_ptr_param->reset(0);

  std::unique_ptr<NonResettable> u_ptr_no_reset;
  u_ptr_no_reset.reset();

  std::shared_ptr<ResettableWithParam> s_ptr_param;
  s_ptr_param.reset();
  s_ptr_param->reset(0);
  s_ptr_param->doSomething();

  std::shared_ptr<NonResettable> s_ptr_no_reset;
  s_ptr_no_reset.reset();

  std::unique_ptr<ResettableWithDefaultParams> u_ptr_default_params;
  u_ptr_default_params.reset(nullptr);
  u_ptr_default_params->reset(1);
  u_ptr_default_params->reset(1, 2.0);

  non_default_reset_ptr<Resettable> non_default_reset_ptr;
  non_default_reset_ptr.reset(new Resettable);
  non_default_reset_ptr->reset();
}

template <typename T>
void TemplatePositiveTest() {
  std::unique_ptr<T> u_ptr;

  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u_ptr).reset();

  std::shared_ptr<T> s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s_ptr = nullptr;
  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s_ptr).reset();
}

template <typename T>
void TemplatNegativeTestTypeWithReset() {
  std::unique_ptr<T> u_ptr;
  u_ptr.reset();
  u_ptr->reset(0);

  std::shared_ptr<T> s_ptr;
  s_ptr.reset();
  s_ptr->reset(0);
}

template <typename T>
void TemplatNegativeTestTypeWithoutReset() {
  std::unique_ptr<T> u_ptr;
  u_ptr.reset();

  std::unique_ptr<T> s_ptr;
  s_ptr.reset();
}

void instantiate() {
  TemplatePositiveTest<Resettable>();
  TemplatePositiveTest<std::unique_ptr<int>>();
  TemplatePositiveTest<std::shared_ptr<int>>();
  TemplatNegativeTestTypeWithReset<ResettableWithParam>();
  TemplatNegativeTestTypeWithoutReset<NonResettable>();
}

struct S {
  void foo() {
    u_ptr.reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider assigning the pointer to 'nullptr' here
    // CHECK-FIXES: u_ptr = nullptr;
    u_ptr->reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
    // CHECK-FIXES: (*u_ptr).reset();
    u_ptr.reset(nullptr);
    u_ptr->doSomething();

    s_ptr.reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider assigning the pointer to 'nullptr' here
    // CHECK-FIXES: s_ptr = nullptr;
    s_ptr->reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
    // CHECK-FIXES: (*s_ptr).reset();
    s_ptr.reset(nullptr);

    ptr.reset();
  }

  std::unique_ptr<Resettable> u_ptr;
  std::unique_ptr<std::shared_ptr<int>> s_ptr;
  std::unique_ptr<NonResettable> ptr;
};


typedef std::unique_ptr<Resettable> TypedefResettableUniquePtr;
typedef std::shared_ptr<Resettable> TypedefResettableSharedPtr;

void TypedefPositive() {
  TypedefResettableUniquePtr u_ptr;
  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u_ptr).reset();

  TypedefResettableSharedPtr s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s_ptr = nullptr;

  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s_ptr).reset();
}

using UsingResettableUniquePtr = std::unique_ptr<Resettable>;
using UsingResettableSharedPtr = std::shared_ptr<Resettable>;

void UsingPositive() {
  UsingResettableUniquePtr u_ptr;
  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u_ptr).reset();

  UsingResettableSharedPtr s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s_ptr = nullptr;

  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s_ptr).reset();
}

template<typename T>
using UsingUniquePtr = std::unique_ptr<T>;
template<typename T>
using UsingSharedPtr = std::shared_ptr<T>;

void UsingTemplatePositive() {
  UsingUniquePtr<Resettable> u_ptr;
  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u_ptr).reset();

  UsingSharedPtr<Resettable> s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s_ptr = nullptr;

  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s_ptr).reset();
}

template<typename T>
void UsingByTemplatePositive() {
  UsingUniquePtr<T> u_ptr;
  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*u_ptr).reset();

  UsingSharedPtr<T> s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: s_ptr = nullptr;

  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*s_ptr).reset();
}

void instantiate2() {
  UsingByTemplatePositive<Resettable>();
}

void NestedUsingPositive() {
  UsingUniquePtr<UsingSharedPtr<TypedefResettableUniquePtr>> nested_ptr;
  nested_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: nested_ptr = nullptr;
  nested_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*nested_ptr).reset();
  (*nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: (*nested_ptr) = nullptr;
  (*nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*(*nested_ptr)).reset();
  (**nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: (**nested_ptr) = nullptr;
  (**nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*(**nested_ptr)).reset();
}

// Check other default pointers and classes.
namespace boost {

template <typename T>
struct shared_ptr {
  T& operator*() const;
  T* operator->() const;
  void reset();
  void reset(T*);
};

} // namespace boost

void PositiveOtherClasses() {
  boost::shared_ptr<Resettable> sh;
  sh.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a smart pointer with pointee that also has a 'reset()' method, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider assigning the pointer to 'nullptr' here
  // CHECK-FIXES: sh = nullptr;
  sh->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: ambiguous call to 'reset()' on a pointee of a smart pointer, prefer more explicit approach
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: consider dereferencing smart pointer to call 'reset' method of the pointee here
  // CHECK-FIXES: (*sh).reset();
}

