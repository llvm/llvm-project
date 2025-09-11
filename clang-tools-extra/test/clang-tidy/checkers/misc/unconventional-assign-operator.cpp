// RUN: %check_clang_tidy %s misc-unconventional-assign-operator %t -- -- -fno-delayed-template-parsing

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
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadArgument const&', 'BadArgument&&' or 'BadArgument'
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
    return *&rhs;
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

// Check that no false positives are issued when using type aliases.
struct TypeAlias {
  using Alias = TypeAlias;
  // This is correct and should not produce any warnings:
  Alias &operator=(const Alias &) { return *this; }

  using AliasRef = Alias &;
  // So is this (assignments from other types are fine):
  AliasRef operator=(int) { return *this; }
};

// Same check as above with typedef instead of using
struct TypeAliasTypedef {
  typedef TypeAliasTypedef Alias;
  Alias &operator=(const Alias &) { return *this; }

  typedef Alias &AliasRef;
  AliasRef operator=(int) { return *this; }
};

// Same check as above for a template class
template <typename T>
struct TemplateTypeAlias {
  using Alias1 = TemplateTypeAlias &;
  using Alias2 = TemplateTypeAlias const &;
  Alias1 operator=(Alias2) { return *this; }

  template <typename U>
  using Alias3 = TemplateTypeAlias<U>;
  Alias3<T> &operator=(int) { return *this; }

  // Using a different type parameter in the return type should give a warning
  Alias3<TypeAlias::Alias> &operator=(double) { return *this; }
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'TemplateTypeAlias&' [misc-unconventional-assign-operator]
};

namespace gh143237 {
template<typename T>
struct TemplateAssignment {
  explicit TemplateAssignment(int) {
  }

  TemplateAssignment& operator=(int n) {
    // No warning
    return *this = TemplateAssignment(n);
  }
};
}

namespace GH153770 {
  struct A;
  struct A {
    A() = default;
    A& operator=(const A&) = default;
  };
} // namespace GH153770
