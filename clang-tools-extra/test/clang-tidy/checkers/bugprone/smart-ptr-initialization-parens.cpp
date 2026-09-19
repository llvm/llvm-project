// RUN: %check_clang_tidy -check-suffix=STRICT -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'true'}}"
// RUN: %check_clang_tidy -check-suffix=PERMISSIVE -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'false'}}"

#include <memory>

void test1() {
  int *i = new int;
  std::shared_ptr<int> p1((i));
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  std::shared_ptr<int> p2((i));
  // CHECK-MESSAGES-PERMISSIVE: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  // CHECK-MESSAGES-STRICT: :[[@LINE-2]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

void test2() {
  std::shared_ptr<int> src;
  std::shared_ptr<int> p1((src.get()));
  // CHECK-MESSAGES-PERMISSIVE: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  // CHECK-MESSAGES-STRICT: :[[@LINE-2]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  std::shared_ptr<int> p2((src.get()));
  // CHECK-MESSAGES-PERMISSIVE: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  // CHECK-MESSAGES-STRICT: :[[@LINE-2]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
}

struct test3 {
  void operator() () {
  std::shared_ptr<int> p1((reinterpret_cast<int*>(this)));
  // CHECK-MESSAGES-PERMISSIVE: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  // CHECK-MESSAGES-STRICT: :[[@LINE-2]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  std::shared_ptr<int> p2((reinterpret_cast<int*>(this)));
  // CHECK-MESSAGES-PERMISSIVE: :[[@LINE-1]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  // CHECK-MESSAGES-STRICT: :[[@LINE-2]]:28: warning: passing a raw pointer 'int *' to 'std::shared_ptr<int>' constructor may cause double deletion
  }
};

