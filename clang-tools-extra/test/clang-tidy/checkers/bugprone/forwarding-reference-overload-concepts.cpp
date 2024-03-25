// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-forwarding-reference-overload %t

template <typename T> constexpr bool just_true = true;

class Test {
public:
  template <typename T> Test(T &&n);
  // CHECK-NOTES: :[[@LINE-1]]:25: warning: constructor accepting a forwarding reference can hide the copy and move constructors

  Test(const Test &rhs);
  // CHECK-NOTES: :[[@LINE-1]]:3: note: copy constructor declared here
};

class Test1 {
public:
  // Guarded with requires expression.
  template <typename T>
  requires requires { just_true<T>; }
  Test1(T &&n);
};

template<typename T>
concept JustTrueConcept = requires { just_true<T>; };

class Test2 {
public:
  // Guarded with concept requirement.
  template <JustTrueConcept T> Test2(T &&n);
};
