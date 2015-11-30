// RUN: %check_clang_tidy %s google-explicit-constructor %t

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

struct A {
  A() {}
  A(int x, int y) {}

  explicit A(void *x) {}
  explicit A(void *x, void *y) {}

  explicit A(const A& a) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}A(const A& a) {}

  A(int x1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit A(int x1) {}

  A(double x2, double y = 3.14) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructors that are callable with a single argument must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit A(double x2, double y = 3.14) {}

  template <typename... T>
  A(T&&... args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructors that are callable with a single argument
  // CHECK-FIXES: {{^  }}explicit A(T&&... args);
};

struct B {
  B(std::initializer_list<int> list1) {}
  B(const std::initializer_list<unsigned> &list2) {}
  B(std::initializer_list<unsigned> &&list3) {}

  explicit B(::std::initializer_list<double> list4) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}B(::std::initializer_list<double> list4) {}

  explicit B(const ::std::initializer_list<char> &list5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}B(const ::std::initializer_list<char> &list5) {}

  explicit B(::std::initializer_list<char> &&list6) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}B(::std::initializer_list<char> &&list6) {}
};

using namespace std;

struct C {
  C(initializer_list<int> list1) {}
  C(const initializer_list<unsigned> &list2) {}
  C(initializer_list<unsigned> &&list3) {}
};

template <typename T>
struct C2 {
  C2(initializer_list<int> list1) {}
  C2(const initializer_list<unsigned> &list2) {}
  C2(initializer_list<unsigned> &&list3) {}

  explicit C2(initializer_list<double> list4) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}C2(initializer_list<double> list4) {}
};

template <typename T>
struct C3 {
  C3(initializer_list<T> list1) {}
  C3(const std::initializer_list<T*> &list2) {}
  C3(::std::initializer_list<T**> &&list3) {}

  template <typename U>
  C3(initializer_list<U> list3) {}
};

struct D {
  template <typename T>
  explicit D(T t) {}
};

template <typename T>
struct E {
  E(T *pt) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors
  // CHECK-FIXES: {{^  }}explicit E(T *pt) {}
  template <typename U>
  E(U *pu) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors
  // CHECK-FIXES: {{^  }}explicit E(U *pu) {}

  explicit E(T t) {}
  template <typename U>
  explicit E(U u) {}
};

void f(std::initializer_list<int> list) {
  D d(list);
  E<decltype(list)> e(list);
  E<int> e2(list);
}
