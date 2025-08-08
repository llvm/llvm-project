// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-scoped-lock %t -- -- -isystem %clang_tidy_headers -fno-delayed-template-parsing

#include <mutex>

void Positive() {
  std::mutex m;
  {
    std::lock_guard<std::mutex> l(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m);
  }

  {
    std::lock_guard<std::mutex> l(m, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(std::adopt_lock, m);
  }

  {
    std::lock_guard<std::mutex> l1(m);
    std::lock_guard<std::mutex> l2(m);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
  }

  {
    std::lock_guard<std::mutex> l1(m), l2(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:40: note: additional 'std::lock_guard' declared here
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
    int a = 0;
    std::lock_guard<std::mutex> l3(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l3(m);
    int b = 0;
    std::lock_guard<std::mutex> l4(m, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l4(std::adopt_lock, m);
  } 
}


std::mutex p_m1;
void PositiveShortFunction() {
  std::lock_guard<std::mutex> l(p_m1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: std::scoped_lock l(p_m1);
}


void PositiveNested() {
  std::mutex m1;
  if (true) {
    std::lock_guard<std::mutex> l(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m1);
    {
      std::lock_guard<std::mutex> l2(m1);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l2(m1);
      {
        std::lock_guard<std::mutex> l3(m1);
        std::lock_guard<std::mutex> l4(m1);
        // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
        // CHECK-MESSAGES: :[[@LINE-2]]:37: note: additional 'std::lock_guard' declared here
      }
      {
        std::lock_guard<std::mutex> l2(m1);
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
        // CHECK-FIXES: std::scoped_lock l2(m1);
      }
    }
  }
  std::lock_guard<std::mutex> l(m1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: std::scoped_lock l(m1);
}


void PositiveInsideArg(std::mutex &m1, std::mutex &m2, std::mutex &m3) {
  std::lock_guard<std::mutex> l1(m1);
  std::lock_guard<std::mutex> l2(m2);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  // CHECK-MESSAGES: :[[@LINE-2]]:31: note: additional 'std::lock_guard' declared here
  int a = 0;
  std::lock_guard<std::mutex> l3(m3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: std::scoped_lock l3(m3);
}


void PositiveInsideConditional() {
  std::mutex m1;
  if (true) {
    std::lock_guard<std::mutex> l1(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m1);
  } else {
    std::lock_guard<std::mutex> l1(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m1);
  }

  while (true) {
    std::lock_guard<std::mutex> l1(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m1);
  }

  for (int i = 0; i < 10; ++i) {
    std::lock_guard<std::mutex> l1(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m1);
  }
}

void PositiveLambda() {
  std::mutex m;
  std::lock_guard<std::mutex> l1(m);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: std::scoped_lock l1(m);
  auto lambda1 = [&]() {
    std::lock_guard<std::mutex> l1(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m);
  };

  std::lock_guard<std::mutex> l3(m);
  std::lock_guard<std::mutex> l4(m);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  // CHECK-MESSAGES: :[[@LINE-2]]:31: note: additional 'std::lock_guard' declared here
  auto lamda2 = [&]() {
    std::lock_guard<std::mutex> l3(m);
    std::lock_guard<std::mutex> l4(m);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
  };

  auto lamda3 = [&]() {
    std::lock(m, m);
    std::lock_guard<std::mutex> l1(m, std::adopt_lock);
    std::lock_guard<std::mutex> l2(m, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
    int a = 0;
    std::lock_guard<std::mutex> l3(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l3(m);
    int b = 0;
    std::lock_guard<std::mutex> l4(m, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l4(std::adopt_lock, m);
  };

  auto lamda4 = [&]() {
    std::lock_guard<std::mutex> l1(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l1(m);
    int a = 0;
    std::lock_guard<std::mutex> l2(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l2(m);
  };
}

template <typename T>
void PositiveTemplated() {
  std::mutex m1, m2, m3;
  {
    std::lock_guard<std::mutex> l(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m1);
  }

  {
    std::lock_guard<std::mutex> l1(m1);
    std::lock_guard<std::mutex> l2(m2);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
  }

  {
    std::lock(m1, m2);
    std::lock_guard<std::mutex> l1(m1, std::adopt_lock);
    std::lock_guard<std::mutex> l2(m2, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:33: note: additional 'std::lock_guard' declared here
    int a = 0;
    std::lock_guard<std::mutex> l3(m3);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l3(m3);
  }
}


template <typename Mutex>
void PositiveTemplatedMutex() {
  Mutex m1, m2, m3;
  {
    std::lock_guard<Mutex> l(m1);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  }

  {
    std::lock_guard<Mutex> l1(m1);
    std::lock_guard<Mutex> l2(m2);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:28: note: additional 'std::lock_guard' declared here
  }

  {
    std::lock(m1, m2);
    std::lock_guard<Mutex> l1(m1, std::adopt_lock);
    std::lock_guard<Mutex> l2(m2, std::adopt_lock);
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-2]]:28: note: additional 'std::lock_guard' declared here
    int a = 0;
    std::lock_guard<Mutex> l3(m3);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  }
}


template <template <typename> typename Lock>
void NegativeTemplate() {
  std::mutex m1, m2;
  {
    Lock<std::mutex> l(m1);
  }

  {
    Lock<std::mutex> l1(m1);
    Lock<std::mutex> l2(m2);
  }
}

void instantiate() {
  NegativeTemplate<std::lock_guard>();
}


struct PositiveClass {
  void Positive() {
    {
      std::lock_guard<std::mutex> l(m1);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l(m1);
    }

    {
      std::lock_guard<std::mutex> l1(m1);
      std::lock_guard<std::mutex> l2(m2);
      // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
      // CHECK-MESSAGES: :[[@LINE-2]]:35: note: additional 'std::lock_guard' declared here
    }

    {
      std::lock(m1, m2);  
      std::lock_guard<std::mutex> l1(m1, std::adopt_lock);
      std::lock_guard<std::mutex> l2(m2, std::adopt_lock);
      // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
      // CHECK-MESSAGES: :[[@LINE-2]]:35: note: additional 'std::lock_guard' declared here
      int a = 0;
      std::lock_guard<std::mutex> l3(m3);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l3(m3);
    }
  }
  
  std::mutex m1;
  std::mutex m2;
  std::mutex m3;
};


template <typename T>
struct PositiveTemplatedClass {
  void Positive() {
    {
      std::lock_guard<std::mutex> l(m1);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l(m1);
    }

    {
      std::lock(m1, m2);  
      std::lock_guard<std::mutex> l1(m1, std::adopt_lock);
      std::lock_guard<std::mutex> l2(m2, std::adopt_lock);
      // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
      // CHECK-MESSAGES: :[[@LINE-2]]:35: note: additional 'std::lock_guard' declared here
      int a = 0;
      std::lock_guard<std::mutex> l3(m3);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l3(m3);
    }
  }

  template <typename... Ts>
  void TemplatedPositive() {
    {
      std::lock_guard<std::mutex> l(m1);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l(m1);
    }

    {
      std::lock(m1, m2);  
      std::lock_guard<std::mutex> l1(m1, std::adopt_lock);
      std::lock_guard<std::mutex> l2(m2, std::adopt_lock);
      // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
      // CHECK-MESSAGES: :[[@LINE-2]]:35: note: additional 'std::lock_guard' declared here
      int a = 0;
      std::lock_guard<std::mutex> l3(m3);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
      // CHECK-FIXES: std::scoped_lock l3(m3);
    }
  }
  
  std::mutex m1;
  std::mutex m2;
  std::mutex m3;
};


template <typename T>
using Lock = std::lock_guard<T>;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
// CHECK-FIXES: using Lock = std::scoped_lock<T>;

using LockM = std::lock_guard<std::mutex>;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
// CHECK-FIXES: using LockM = std::scoped_lock<std::mutex>;

typedef std::lock_guard<std::mutex> LockDef;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
// CHECK-FIXES: typedef std::scoped_lock<std::mutex> LockDef;


void PositiveUsingDecl() {
  using std::lock_guard;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: using std::scoped_lock;

  using LockMFun = std::lock_guard<std::mutex>;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: using LockMFun = std::scoped_lock<std::mutex>;

  typedef std::lock_guard<std::mutex> LockDefFun;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: typedef std::scoped_lock<std::mutex> LockDefFun;
}

template <typename T>
void PositiveUsingDeclTemplate() {
  using std::lock_guard;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: using std::scoped_lock;

  std::mutex m;
  lock_guard<std::mutex> l(m);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: std::scoped_lock l(m);

  using LockFunT = std::lock_guard<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: using LockFunT = std::scoped_lock<T>;

  using LockMFunT = std::lock_guard<std::mutex>;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: using LockMFunT = std::scoped_lock<std::mutex>;

  typedef std::lock_guard<std::mutex> LockDefFunT;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  // CHECK-FIXES: typedef std::scoped_lock<std::mutex> LockDefFunT;
}

void PositiveInUsingTypedefs() {
  std::mutex m;

  {
    Lock<std::mutex> l(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m);
  }

  {
    LockM l(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m);
  }

  {
    LockDef l(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l(m);
  }

  {
    std::lock(m, m);
    Lock<std::mutex> l1(m, std::adopt_lock);
    LockM l2(m, std::adopt_lock);
    LockDef l3(m), l4(m);
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-3]]:11: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-3]]:13: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-4]]:20: note: additional 'std::lock_guard' declared here
    int a = 0;
    LockDef l5(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
    // CHECK-FIXES: std::scoped_lock l5(m);
  }
}

template <typename Mutex>
void PositiveInUsingTypedefsTemplated() {
  Mutex m;

  {
    Lock<Mutex> l(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  }

  {
    std::lock(m, m);
    Lock<Mutex> l1(m, std::adopt_lock);
    LockM l2(m, std::adopt_lock);
    LockDef l3(m), l4(m);
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
    // CHECK-MESSAGES: :[[@LINE-3]]:11: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-3]]:13: note: additional 'std::lock_guard' declared here
    // CHECK-MESSAGES: :[[@LINE-4]]:20: note: additional 'std::lock_guard' declared here
    int a = 0;
    LockDef l5(m);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::scoped_lock' instead of 'std::lock_guard'
  }
}

// Non-STD lock_guard.
template <typename Mutex>
struct lock_guard {
  lock_guard(Mutex &m) { }
  lock_guard(const lock_guard& ) = delete;
};

void NegativeNonStdLockGuard() {
  std::mutex m;
  {
    lock_guard<std::mutex> l(m);
  }

  {
    lock_guard<std::mutex> l1(m);
    lock_guard<std::mutex> l2(m);
  }
}
