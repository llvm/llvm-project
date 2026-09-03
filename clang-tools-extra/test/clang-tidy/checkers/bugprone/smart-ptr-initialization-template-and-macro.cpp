// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'true'}}"
// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'false'}}"

#include <memory>

struct A {
  int x;
};

template<typename T>
struct test_shared_ptr_constructor_template {
  void operator() () {
  T a(reinterpret_cast<A*>(this));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
};

int a = (test_shared_ptr_constructor_template<std::shared_ptr<A>>()(), 0);

#define SHARED_PTR_A std::shared_ptr<A>
#define UNIQUE_PTR_A std::unique_ptr<A>

struct test_shared_ptr_constructor_macro1 {
  void operator() () {
  SHARED_PTR_A a(reinterpret_cast<A*>(this));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
};

struct test_unique_ptr_constructor_macro1 {
  void operator() () {
  UNIQUE_PTR_A b(reinterpret_cast<A*>(this));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
};

#define THIS reinterpret_cast<A*>(this)

struct test_shared_ptr_constructor_macro2 {
  void operator() () {
  std::shared_ptr<A> a(THIS);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
};

struct test_unique_ptr_constructor_macro2 {
  void operator() () {
  std::unique_ptr<A> b(THIS);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
};

#define SHARED_PTR_THE_WHOLE_STATEMENT_IN_MACRO std::shared_ptr<A> a(reinterpret_cast<A*>(this));
#define UNIQUE_PTR_THE_WHOLE_STATEMENT_IN_MACRO std::unique_ptr<A> b(reinterpret_cast<A*>(this));

struct test_shared_ptr_constructor_macro3 {
  void operator() () {
  SHARED_PTR_THE_WHOLE_STATEMENT_IN_MACRO
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
};

struct test_unique_ptr_constructor_macro3 {
  void operator() () {
  UNIQUE_PTR_THE_WHOLE_STATEMENT_IN_MACRO
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
};

#define COMPLICATED_SOURCE_LOCATION_FOR_A (reinterpret_cast<A*>(
#define COMPLICATED_SOURCE_LOCATION_FOR_B (reinterpret_cast<A*>(

struct test_shared_ptr_constructor_macro4 {
  void operator() () {
  std::shared_ptr<A> COMPLICATED_SOURCE_LOCATION_FOR_A this));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
};

struct test_unique_ptr_constructor_macro5 {
  void operator() () {
  std::unique_ptr<A> COMPLICATED_SOURCE_LOCATION_FOR_B this));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
};
