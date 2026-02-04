// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-scoped-lock %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-scoped-lock.WarnOnSingleLocks: false}}" \
// RUN:   -- -isystem %clang_tidy_headers -fno-delayed-template-parsing

#include <mutex>

void Positive() {
  std::mutex m;

  {
    std::lock_guard<std::mutex> l1(m);
    std::lock_guard<std::mutex> l2(m);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
  }

  {
    std::lock_guard<std::mutex> l1(m), l2(m), l3(m);
    std::lock_guard<std::mutex> l4(m);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-3]]:40: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-4]]:47: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-4]]:33: note: additional 'std::lock_guard' declared here
  }
  
  { 
    std::lock(m, m);
    std::lock_guard<std::mutex> l1(m, std::adopt_lock);
    std::lock_guard<std::mutex> l2(m, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
  } 
}

void Negative() {
  std::mutex m;
  {
    std::lock_guard<std::mutex> l(m);
  }

  {
    std::lock_guard<std::mutex> l(m, std::adopt_lock);
  }

  { 
    std::lock_guard<std::mutex> l3(m);
    int a = 0;
    std::lock_guard<std::mutex> l4(m, std::adopt_lock);
  }
}

void PositiveInsideArg(std::mutex &m1, std::mutex &m2, std::mutex &m3) {
  std::lock_guard<std::mutex> l1(m1);
  std::lock_guard<std::mutex> l2(m2);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  // CHECK-MESSAGES: :[[@LINE-2]]:31: note: additional 'std::lock_guard' declared here
}


void NegativeInsideArg(std::mutex &m1, std::mutex &m2, std::mutex &m3) {
  std::lock_guard<std::mutex> l3(m3);
}

template <typename T>
void PositiveTemplated() {
  std::mutex m1, m2;
  
  std::lock_guard<std::mutex> l1(m1);
  std::lock_guard<std::mutex> l2(m2);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  // CHECK-MESSAGES: :[[@LINE-2]]:31: note: additional 'std::lock_guard' declared here  
}

template <typename T>
void NegativeTemplated() {
  std::mutex m1, m2, m3;
  std::lock_guard<std::mutex> l(m1);
}

template <typename Mutex>
void PositiveTemplatedMutex() {
  Mutex m1, m2;

  std::lock_guard<Mutex> l1(m1);
  std::lock_guard<Mutex> l2(m2);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: additional 'std::lock_guard' declared here
}

template <typename Mutex>
void NegativeTemplatedMutex() {
  Mutex m1;
  std::lock_guard<Mutex> l(m1);
}

struct NegativeClass {
  void Negative() {
    std::lock_guard<std::mutex> l(m1);
  }
  
  std::mutex m1;
};
