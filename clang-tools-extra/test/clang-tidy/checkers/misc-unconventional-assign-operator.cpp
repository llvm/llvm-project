// RUN: %check_clang_tidy %s misc-unconventional-assign-operator %t -- -- -isystem %S/Inputs/Headers -fno-delayed-template-parsing

namespace std {
template <typename T>
struct remove_reference { typedef T type; };
template <typename T>
struct remove_reference<T &> { typedef T type; };
template <typename T>
struct remove_reference<T &&> { typedef T type; };
template <typename T>
typename remove_reference<T>::type &&move(T &&t);
}


struct Good {
  Good& operator=(const Good&);
  Good& operator=(Good&&);

  // Assign from other types is fine too.
  Good& operator=(int);
};

struct AlsoGood {
  // By value is also fine.
  AlsoGood& operator=(AlsoGood);
};

struct BadReturnType {
  void operator=(const BadReturnType&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'BadReturnType&' [misc-unconventional-assign-operator]
  const BadReturnType& operator=(BadReturnType&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
  void operator=(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
};

struct BadReturnType2 {
  BadReturnType2&& operator=(const BadReturnType2&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
  int operator=(BadReturnType2&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
};

struct BadArgument {
  BadArgument& operator=(BadArgument&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadArgument const&', 'BadArgument&&' or 'BadArgument'
  BadArgument& operator=(const BadArgument&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadAr
};

struct BadModifier {
  BadModifier& operator=(const BadModifier&) const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should not be marked 'const'
};

struct Deleted {
  // We don't check the return value of deleted operators.
  void operator=(const Deleted&) = delete;
  void operator=(Deleted&&) = delete;
};

class Private {
  // We don't check the return value of private operators.
  // Pre-C++11 way of disabling assignment.
  void operator=(const Private &);
};

struct Virtual {
  virtual Virtual& operator=(const Virtual &);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should not be marked 'virtual'
};

class BadReturnStatement {
  int n;

public:
  BadReturnStatement& operator=(BadReturnStatement&& rhs) {
    n = std::move(rhs.n);
    return rhs;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: operator=() should always return '*this'
  }

  // Do not check if return type is different from '&BadReturnStatement'
  int operator=(int i) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
    n = i;
    return n;
  }
};

namespace pr31531 {
enum E { e };
// This declaration makes the 'return *this' below have an unresolved operator
// in the class template, but not in an instantiation.
E operator*(E, E);

template <typename>
struct UnresolvedOperator {
  UnresolvedOperator &operator=(const UnresolvedOperator &) { return *this; }
};

UnresolvedOperator<int> UnresolvedOperatorInt;

template <typename>
struct Template {
  Template &operator=(const Template &) { return this; }
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: operator=() should always return '*this'
};

Template<int> TemplateInt;
}

struct AssignmentCallAtReturn {
  AssignmentCallAtReturn &returnThis() {
    return *this;
  }
  AssignmentCallAtReturn &operator=(int rhs) {
    return *this;
  }
  AssignmentCallAtReturn &operator=(char rhs) {
    // Allow call to assignment from other type.
    return (*this = static_cast<int>(rhs));
  }
  AssignmentCallAtReturn &operator=(float rhs) {
    // Do not allow calls to other functions.
    return returnThis();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: operator=() should always return '*this'
  }
};
