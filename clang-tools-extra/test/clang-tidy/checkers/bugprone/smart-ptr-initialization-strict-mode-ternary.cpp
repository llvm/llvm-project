// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'true'}}"

#include <memory>

struct A {
  int x;
};

A& getA();
A* getAPtr();
bool flag = false;
bool flag2 = false;

void test_new_expression_trivial_ok() {
  std::shared_ptr<A> a(flag ? new A{100} : new A{200});
  std::unique_ptr<A> b(flag ? new A{200} : new A{100});
}

void test_new_expression_ok() {
  std::shared_ptr<A> a(flag ? new A() : nullptr);
  std::unique_ptr<A> b(flag ? nullptr : new A());
}

void test_new_expression_with_zero_ok() {
  std::shared_ptr<A> a(flag ? new A() : 0);
  std::unique_ptr<A> b(flag ? 0 : new A());
}

void test_release_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A> a(flag ? p1.release() : nullptr);
  std::shared_ptr<A> b(flag ? p1.release() : nullptr);
}

void test_release_cast_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A> a(flag ? static_cast<A*>(p1.release()) : 0);
  std::shared_ptr<A> b(flag ? static_cast<A*>(p1.release()) : 0);
}


void test_smaprt_ptr_constructor() {
  std::shared_ptr<A> a(flag ? &getA() : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(flag ? &getA() : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

struct NoopDeleter {
    void operator() (A* p) {}
};

void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0(flag ? &getA() : new A);
  std::unique_ptr<A, decltype(noop_deleter)> p1(flag ? &getA() : nullptr, noop_deleter);
  std::shared_ptr<A> p2(flag ? &getA() : 0, noop_deleter);
}




void test_new_expression_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(flag ? new A() : nullptr);
  std::unique_ptr<A> b;
  b.reset(flag ? nullptr : new A());
}

void test_release_reset_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A> a;
  a.reset(flag ? p1.release() : nullptr);
  std::unique_ptr<A> b;
  b.reset(flag ? p1.release() : nullptr);
}

void test_release_reset_cast_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A> a;
  a.reset(static_cast<A*>(flag ? p1.release() : nullptr));
  std::shared_ptr<A> b;
  b.reset(static_cast<A*>(flag ? p1.release() : nullptr));
}

void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0;
  p0.reset(flag ? &getA() : nullptr);
  std::unique_ptr<A, decltype(noop_deleter)> p1;
  p1.reset(flag ? &getA() : nullptr);
  std::shared_ptr<A> p2;
  // FIXME: mock shared_ptr must support reset with custom deleter
  // p2.reset(flag ? &getA() : nullptr, noop_deleter);
}

void test_smaprt_ptr_reset() {
  std::shared_ptr<A> a;
  a.reset(flag ? &getA() : nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  std::unique_ptr<A> b;
  b.reset(flag ? &getA() : nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' reset method may cause double deletion
}

// ===== NESTED TERNARY OPERATORS IN CONSTRUCTORS =====

void test_nested_ternary_new_expression_ok() {
  std::shared_ptr<A> a(flag ? (flag2 ? new A{100} : new A{200}) : new A{300});
  std::unique_ptr<A> b(flag ? new A{100} : (flag2 ? new A{200} : new A{300}));
}

void test_nested_ternary_with_nullptr_ok() {
  std::shared_ptr<A> a(flag ? (flag2 ? new A() : nullptr) : new A());
  std::unique_ptr<A> b(flag ? new A() : (flag2 ? nullptr : new A()));
}

void test_nested_ternary_with_zero_ok() {
  std::shared_ptr<A> a(flag ? (flag2 ? new A() : 0) : new A());
  std::unique_ptr<A> b(flag ? new A() : (flag2 ? 0 : new A()));
}

void test_nested_ternary_release_ok(std::unique_ptr<A> p1, std::unique_ptr<A> p2) {
  std::unique_ptr<A> a(flag ? (flag2 ? p1.release() : p2.release()) : nullptr);
  std::shared_ptr<A> b(flag ? nullptr : (flag2 ? p1.release() : p2.release()));
}

void test_nested_ternary_release_cast_ok(std::unique_ptr<A> p1, std::unique_ptr<A> p2) {
  std::unique_ptr<A> a(flag ? static_cast<A*>(flag2 ? p1.release() : p2.release()) : 0);
  std::shared_ptr<A> b(flag ? 0 : static_cast<A*>(flag2 ? p1.release() : p2.release()));
}

void test_nested_ternary_mixed_release_new_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A> a(flag ? (flag2 ? p1.release() : new A()) : nullptr);
  std::shared_ptr<A> b(flag ? nullptr : (flag2 ? new A() : p1.release()));
}

// ===== NESTED TERNARY OPERATORS WITH DANGEROUS RAW POINTERS =====

void test_nested_ternary_dangerous_raw_ptr() {
  std::shared_ptr<A> a(flag ? (flag2 ? &getA() : new A) : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  
  std::unique_ptr<A> b(flag ? new A : (flag2 ? new A : &getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
  
  std::shared_ptr<A> c(flag ? (flag2 ? new A : &getA()) : (flag2 ? &getA() : new A));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_nested_ternary_dangerous_getAPtr() {
  std::shared_ptr<A> a(flag ? (flag2 ? getAPtr() : new A) : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  
  std::unique_ptr<A> b(flag ? new A : (flag2 ? new A : getAPtr()));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

// ===== NESTED TERNARY OPERATORS WITH CAST =====

void test_nested_ternary_dangerous_cast() {
  std::shared_ptr<A> a(flag ? static_cast<A*>(flag2 ? &getA() : new A) : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  
  std::unique_ptr<A> b(flag ? new A : static_cast<A*>(flag2 ? new A : &getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}

// ===== NESTED TERNARY OPERATORS IN RESET =====

void test_nested_ternary_reset_new_ok() {
  std::shared_ptr<A> a;
  a.reset(flag ? (flag2 ? new A() : new A{100}) : new A());
  
  std::unique_ptr<A> b;
  b.reset(flag ? new A() : (flag2 ? new A{200} : new A()));
}

void test_nested_ternary_reset_with_nullptr_ok() {
  std::shared_ptr<A> a;
  a.reset(flag ? (flag2 ? new A() : nullptr) : new A());
  
  std::unique_ptr<A> b;
  b.reset(flag ? new A() : (flag2 ? nullptr : new A()));
}

void test_nested_ternary_reset_release_ok(std::unique_ptr<A> p1, std::unique_ptr<A> p2) {
  std::unique_ptr<A> a;
  a.reset(flag ? (flag2 ? p1.release() : p2.release()) : nullptr);
  
  std::shared_ptr<A> b;
  b.reset(flag ? nullptr : (flag2 ? p1.release() : p2.release()));
}

void test_nested_ternary_reset_dangerous_raw_ptr() {
  std::shared_ptr<A> a;
  a.reset(flag ? (flag2 ? &getA() : new A) : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  
  std::unique_ptr<A> b;
  b.reset(flag ? new A : (flag2 ? new A : &getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' reset method may cause double deletion
}

void test_nested_ternary_reset_dangerous_getAPtr() {
  std::shared_ptr<A> a;
  a.reset(flag ? (flag2 ? getAPtr() : new A) : new A);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  
  std::unique_ptr<A> b;
  b.reset(flag ? new A : (flag2 ? getAPtr() : new A));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' reset method may cause double deletion
}

// ===== NESTED TERNARY OPERATORS WITH CAST IN RESET =====

void test_nested_ternary_reset_cast_ok(std::unique_ptr<A> p1, std::unique_ptr<A> p2) {
  std::unique_ptr<A> a;
  a.reset(static_cast<A*>(flag ? (flag2 ? p1.release() : p2.release()) : nullptr));
  
  std::shared_ptr<A> b;
  b.reset(static_cast<A*>(flag ? nullptr : (flag2 ? p1.release() : p2.release())));
}

void test_nested_ternary_reset_cast_dangerous() {
  std::shared_ptr<A> a;
  a.reset(static_cast<A*>(flag ? (flag2 ? &getA() : new A) : new A));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
  
  std::unique_ptr<A> b;
  b.reset(static_cast<A*>(flag ? new A : (flag2 ? new A : &getA())));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' reset method may cause double deletion
}

// ===== NESTED TERNARY OPERATORS WITH A CUSTOM DELETER =====

void test_nested_ternary_custom_deleter_ok() {
  auto noop_deleter = [](A* p) { };
  
  std::unique_ptr<A, NoopDeleter> p0(flag ? (flag2 ? &getA() : new A) : new A);
  std::unique_ptr<A, decltype(noop_deleter)> p1(
    flag ? new A : (flag2 ? &getA() : nullptr), noop_deleter);
  std::shared_ptr<A> p2(flag ? (flag2 ? &getA() : 0) : new A, noop_deleter);
}

void test_nested_ternary_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) { };
  
  std::unique_ptr<A, NoopDeleter> p0;
  p0.reset(flag ? (flag2 ? &getA() : new A) : new A);
  
  std::unique_ptr<A, decltype(noop_deleter)> p1;
  p1.reset(flag ? new A : (flag2 ? &getA() : nullptr));
  
  std::shared_ptr<A> p2;
  // p2.reset(flag ? (flag2 ? &getA() : nullptr) : new A, noop_deleter);
  // FIXME: mock shared_ptr must support reset with custom deleter
}

// ===== DEEP NESTING (3+ levels) =====

void test_deep_nested_ternary_ok() {
  bool flag3 = false;
  
  std::shared_ptr<A> a(
    flag ? (flag2 ? (flag3 ? new A{1} : new A{2}) : new A{3}) : new A{4}
  );
  
  std::unique_ptr<A> b(
    flag ? new A{1} : (flag2 ? (flag3 ? new A{2} : new A{3}) : new A{4})
  );
}

void test_deep_nested_ternary_dangerous() {
  bool flag3 = false;
  
  std::shared_ptr<A> a(
    flag ? (flag2 ? (flag3 ? &getA() : new A) : new A) : new A
  );
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
  
  std::unique_ptr<A> b(
    flag ? new A : (flag2 ? new A : (flag3 ? new A : &getA()))
  );
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A>' constructor may cause double deletion
}
