// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>

struct A {
  int x;
};

A& getA();
A* getAPtr();

#define SHARED_PTR_A std::shared_ptr<A>
#define UNIQUE_PTR_A std::unique_ptr<A>

void test_shared_ptr_constructor_macro1() {
  SHARED_PTR_A a(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor_macro1() {
  UNIQUE_PTR_A b(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

#define GET_REFERENCE_TO_GETA_RESULT &getA()

void test_shared_ptr_constructor_macro2() {
  std::shared_ptr<A> a(GET_REFERENCE_TO_GETA_RESULT);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor_macro2() {
  std::unique_ptr<A> b(GET_REFERENCE_TO_GETA_RESULT);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

#define SHARED_PTR_THE_WHOLE_STATEMENT_IN_MACRO std::shared_ptr<A> a(&getA());
#define UNIQUE_PTR_THE_WHOLE_STATEMENT_IN_MACRO std::unique_ptr<A> b(&getA());

void test_shared_ptr_constructor_macro3() {
  SHARED_PTR_THE_WHOLE_STATEMENT_IN_MACRO
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor_macro3() {
  UNIQUE_PTR_THE_WHOLE_STATEMENT_IN_MACRO
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

#define COMPLICATED_SOURCE_LOCATION_FOR_A a(&
#define COMPLICATED_SOURCE_LOCATION_FOR_B b(&

void test_shared_ptr_constructor_macro4() {
  std::shared_ptr<A> COMPLICATED_SOURCE_LOCATION_FOR_A getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor_macro5() {
  std::unique_ptr<A> COMPLICATED_SOURCE_LOCATION_FOR_B getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}
