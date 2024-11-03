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
