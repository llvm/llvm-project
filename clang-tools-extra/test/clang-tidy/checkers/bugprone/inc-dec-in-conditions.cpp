// RUN: %check_clang_tidy %s bugprone-inc-dec-in-conditions %t

template<typename T>
struct Iterator {
  Iterator operator++(int);
  Iterator operator--(int);
  Iterator& operator++();
  Iterator& operator--();
  T operator*();
  bool operator==(Iterator) const;
  bool operator!=(Iterator) const;
};

template<typename T>
struct Container {
  Iterator<T> begin();
  Iterator<T> end();
};

bool f(int x) {
  return (++x != 5 or x == 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: incrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool f2(int x) {
  return (x++ != 5 or x == 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: incrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool c(Container<int> x) {
  auto it = x.begin();
  return (it++ != x.end() and *it);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: incrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool c2(Container<int> x) {
  auto it = x.begin();
  return (++it != x.end() and *it);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: incrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool d(int x) {
  return (--x != 5 or x == 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: decrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool d2(int x) {
  return (x-- != 5 or x == 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: decrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool g(Container<int> x) {
  auto it = x.begin();
  return (it-- != x.end() and *it);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: decrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool g2(Container<int> x) {
  auto it = x.begin();
  return (--it != x.end() and *it);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: decrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

bool doubleCheck(Container<int> x) {
  auto it = x.begin();
  auto it2 = x.begin();
  return (--it != x.end() and ++it2 != x.end()) and (*it == *it2);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: decrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
  // CHECK-MESSAGES: :[[@LINE-2]]:31: warning: incrementing and referencing a variable in a complex condition can cause unintended side-effects due to C++'s order of evaluation, consider moving the modification outside of the condition to avoid misunderstandings [bugprone-inc-dec-in-conditions]
}

namespace PR85838 {
  void test()
  {
    auto foo = 0;
    auto bar = 0;
    if (++foo < static_cast<decltype(foo)>(bar)) {}
    if (static_cast<decltype(++foo)>(bar) < foo) {}
  }
}

namespace GH163913 {
  void lambdaWithIncrement(int size) {
    if ([](int n) {
        ++n;
        return n;
    }(size) < 42) {}
  }

  void lambdaWithDecrement(int size) {
    if ([](int n) {
        --n;
        return n;
    }(size) < 42) {}
  }

  template <typename Func>
  auto eval(Func&& fn) -> decltype(fn()) {
    return fn();
  }

  void lambdaWithForLoop(int size) {
    if (eval([&] {
        int result = 0;
        for (int i = 0; i < size; i++) {
            result += i;
        }
        return result;
    }) < 42) {}
  }

  void outsideLambda(int size) {
    if ([](int n) {
        --n;
        return n;
    }(size) < ++size) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: incrementing and referencing a variable in a complex condition
    if (++size > [](int n) {
        --n;
        return n;
    }(size)) {}
    // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: incrementing and referencing a variable in a complex condition
  }
}
