// RUN: %check_clang_tidy %s bugprone-smartptr-reset-ambiguous-call %t --fix-notes --

namespace std {

template <typename T>
struct unique_ptr {
  T& operator*() const;
  T* operator->() const;
  void reset(T* p = nullptr);
};

template <typename T>
struct shared_ptr {
  T& operator*() const;
  T* operator->() const;
  void reset();
  void reset(T*);
};

} // namespace std

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
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: u = nullptr;
  u->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*u).reset();

  std::shared_ptr<Resettable> s;
  s.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: s = nullptr;
  s->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*s).reset();

  std::unique_ptr<std::unique_ptr<int>> uu_ptr;
  uu_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: uu_ptr = nullptr;
  uu_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*uu_ptr).reset();

  std::unique_ptr<std::shared_ptr<int>> su_ptr;
  su_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: su_ptr = nullptr;
  su_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*su_ptr).reset();

  std::unique_ptr<ResettableWithDefaultParams> rd;
  rd.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: rd = nullptr;
  rd->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*rd).reset();

  std::unique_ptr<std::shared_ptr<std::unique_ptr<Resettable>>> nested_ptr;
  nested_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: nested_ptr = nullptr;
  nested_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*nested_ptr).reset();
  (*nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: (*nested_ptr) = nullptr;
  (*nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*(*nested_ptr)).reset();
  (**nested_ptr).reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: (**nested_ptr) = nullptr;
  (**nested_ptr)->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
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
}

template <typename T>
void TemplatePositiveTest() {
  std::unique_ptr<T> u_ptr;

  u_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: u_ptr = nullptr;
  u_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
  // CHECK-FIXES: (*u_ptr).reset();

  std::shared_ptr<T> s_ptr;
  s_ptr.reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: assign the pointer to 'nullptr'
  // CHECK-FIXES: s_ptr = nullptr;
  s_ptr->reset();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: use dereference to call 'reset' method of the pointee
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
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: assign the pointer to 'nullptr'
    // CHECK-FIXES: u_ptr = nullptr;
    u_ptr->reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: use dereference to call 'reset' method of the pointee
    // CHECK-FIXES: (*u_ptr).reset();
    u_ptr.reset(nullptr);
    u_ptr->doSomething();

    s_ptr.reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: be explicit when calling 'reset()' on a smart pointer with a pointee that has a 'reset()' method
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: assign the pointer to 'nullptr'
    // CHECK-FIXES: s_ptr = nullptr;
    s_ptr->reset();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: be explicit when calling 'reset()' on a pointee of a smart pointer
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: use dereference to call 'reset' method of the pointee
    // CHECK-FIXES: (*s_ptr).reset();
    s_ptr.reset(nullptr);

    ptr.reset();
  }

  std::unique_ptr<Resettable> u_ptr;
  std::unique_ptr<std::shared_ptr<int>> s_ptr;
  std::unique_ptr<NonResettable> ptr;
};
