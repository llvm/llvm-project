// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t

#include <memory>
#include <string>
#include <utility>

struct A {
  int* val;
};

A* getAPtr();

void test_new_expression_ok() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
}

void test_new_expression_cast_ok() {
  A* first = static_cast<A*>(new A());
  A* second = static_cast<A*>(new A());
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
}

void test_basic_double_two_conditions_ok(bool cond) {
  int* a = new int(42);
  if (cond) {
    std::shared_ptr<int> p1(a);
  } else {
    std::shared_ptr<int> p2(a);
  }
}

void test_basic_no_double_ownership() {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);
  a = nullptr;
  std::shared_ptr<int> p2(a);
}

void test_reassignment_valid() {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);

  a = new int(43);  // Reassign
  std::shared_ptr<int> p2(a);  // OK - new memory

  int* b = new int(44);
  std::shared_ptr<int> p3(b);  // OK
}

void test_make_shared() {
  auto p1 = std::make_shared<int>(42);
  auto p2 = p1;
}

void test_move_smart_ptr() {
  int* a = new int(42);
  std::shared_ptr<int> s1(a);
  std::shared_ptr<int> s2(std::move(s1));
  int* b = new int(42);
  std::shared_ptr<int> u1(b);
  std::shared_ptr<int> u2(std::move(u1));
}

void test_new_expression_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

using ptr_a_t = A*;
using shared_ptr_a_t = std::shared_ptr<A>;
using unique_ptr_a_t = std::unique_ptr<A>;

void test_new_expression_with_aliases() {
  ptr_a_t first = new A();
  ptr_a_t second = new A();
  shared_ptr_a_t a(first);
  shared_ptr_a_t a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: passing a raw pointer 'ptr_a_t' (aka 'A *') to 'shared_ptr_a_t' (aka 'std::shared_ptr<A>') constructor may cause double deletion
  unique_ptr_a_t b(second);
  unique_ptr_a_t b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: passing a raw pointer 'ptr_a_t' (aka 'A *') to 'unique_ptr_a_t' (aka 'std::unique_ptr<A>') constructor may cause double deletion
}

void test_new_expression_fail_in_different_scopes() {
  A* first = new A();
  A* second = new A();
  {
  std::shared_ptr<A> a(first);
  }
  {
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
  {
  std::unique_ptr<A> b(second);
  }
  {
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
}

void test_new_expression_fail_in_different_scopes2() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  {
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
  
  std::unique_ptr<A> b(second);
  {
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
}

void test_new_expression_ok_in_scope() {
  {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
  }
}

void test_new_expression_fail_in_scope() {
  {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
}

struct B {

void test_new_expression_ok_as_method() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
}

void test_new_expression_fail_as_method() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

};

void test_reassignment_double_ownership() {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);

  a = new int(43);  // Reassign
  std::shared_ptr<int> p2(a);
  std::shared_ptr<int> p3(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_reset_to_nullptr() {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);
  a = nullptr;  //releasing the owning

  int* b = new int(43);
  std::shared_ptr<int> p2(b);  // OK - new memory
  // no warnings
}

void test_branch(bool cond) {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);

  if (cond) {
    std::shared_ptr<int> p2(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  }
}

void test_loop() {
  int* a = new int(42);
  std::shared_ptr<int> p1(a);

  for (int i = 0; i < 10; ++i) {
    std::shared_ptr<int> p2(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  }
}

void test_multiple_variables() {
  int* a = new int(42);
  int* b = new int(43);

  std::shared_ptr<int> p1(a);
  std::shared_ptr<int> p2(b);
  std::shared_ptr<int> p3(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_function_parameter(int* a) {
  std::shared_ptr<int> p1(a);
  std::shared_ptr<int> p2(a);  // We don't know where this memory came from, but it doesn't matter anymore, since it will be freed at least twice
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_argument(int* a) {
  a = new int(42);
  std::shared_ptr<int> p1(a);
  // Это должно вызвать предупреждение
  std::shared_ptr<int> p2(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_complex_case() {
  int* a = new int(1);
  std::shared_ptr<int> p1(a);

  a = new int(2);  // Reassign
  std::shared_ptr<int> p2(a);  // OK

  a = new int(3);  // Reassign again
  std::shared_ptr<int> p3(a);  // OK
  std::shared_ptr<int> p4(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_nested_function() {
  auto lambda = []() {
    int* a = new int(42);
    std::shared_ptr<int> p1(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
    std::shared_ptr<int> p2(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  };
  lambda();
}

void test_nested_function2() {
  int* a = new int(42);
  auto lambda = [&]() {
    std::shared_ptr<int> p1(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
    std::shared_ptr<int> p2(a);
  };
  lambda();
}

void test_inside_structure() {
  A a;
  a.val = new int(42);
  std::shared_ptr<int> p1(a.val);
  std::shared_ptr<int> p2(a.val);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test_inside_structure_and_argument(A& a) {
  a.val = new int(42);
  std::shared_ptr<int> p1(a.val);
  std::shared_ptr<int> p2(a.val);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

class test_inside_a_class {
  int* a = nullptr;
public:
  void operator() () {
    a = new int(42);
    std::shared_ptr<int> p1(a);
    std::shared_ptr<int> p2(a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  }
};

class test_inside_a_class_with_this {
  int* a = nullptr;
public:
  void operator() () {
    this->a = new int(42);
    std::shared_ptr<int> p1(this->a);
    std::shared_ptr<int> p2(this->a);
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  }
};

struct Inner {
  int *P = nullptr;
};
struct Outer {
  Inner In;
};

void double_wrap_nested_fields(Outer &O) {
  O.In.P = new int;
  std::shared_ptr<int> First(O.In.P);
  std::shared_ptr<int> Second(O.In.P);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

// test_new_expression_ok_in_global
// FIXME: support it
/*
A* first = new A();
A* second = new A();
std::shared_ptr<A> a(first);
std::unique_ptr<A> b(second);

// test_new_expression_fail_in_global
A* first2 = new A();
A* second2 = new A();
std::shared_ptr<A> a1(first2);
std::shared_ptr<A> a2(first2);
std::unique_ptr<A> b1(second2);
std::unique_ptr<A> b2(second2);
*/


void test_new_expression_reset_ok() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a;
  a.reset(first);
  std::unique_ptr<A> b;
  b.reset(second);
}

void test_basic_double_two_conditions_ok(bool cond, std::shared_ptr<int> p1, std::shared_ptr<int> p2) {
  int* a = new int(42);
  if (cond) {
     p1.reset(a);
  } else {
     p2.reset(a);
  }
}

void test_basic_no_double_ownership(std::shared_ptr<int> p1, std::shared_ptr<int> p2) {
  int* a = new int(42);
  p1.reset(a);
  a = nullptr;
  p2.reset(a);
}

void test_new_expression_reset_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a;
  a.reset(first);
  std::shared_ptr<A> a2;
  a2.reset(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  std::unique_ptr<A> b;
  b.reset(second);
  std::unique_ptr<A> b2;
  b2.reset(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' reset method may cause double deletion
}

void test_new_expression_reset_fail_with_aliases() {
  ptr_a_t first = new A();
  ptr_a_t second = new A();
  shared_ptr_a_t a;
  a.reset(first);
  shared_ptr_a_t a2;
  a2.reset(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'ptr_a_t' (aka 'A *') to 'shared_ptr_a_t' (aka 'std::shared_ptr<A>') reset method may cause double deletion
  unique_ptr_a_t b;
  b.reset(second);
  unique_ptr_a_t b2;
  b2.reset(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'ptr_a_t' (aka 'A *') to 'unique_ptr_a_t' (aka 'std::unique_ptr<A>') reset method may cause double deletion
}

void test_new_expression_crossed_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2;
  a2.reset(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  std::unique_ptr<A> b;
  b.reset(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

bool can_take(std::shared_ptr<A> a);
void take(std::shared_ptr<A> a);

void test_discowered_in_wild(std::shared_ptr<A> a, std::unique_ptr<A> b) {
  if (can_take(a)) {
    take(a);
  }
}

void test_new_expression_not_only_smartpointers() {
  char* first = new char();
  char* second = new char();
  std::shared_ptr<char> a(first);
  std::string a2(first);
  std::unique_ptr<char> b(second);
  std::string b2(second);
}
